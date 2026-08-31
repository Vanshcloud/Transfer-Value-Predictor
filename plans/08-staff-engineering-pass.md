# Phase 18 — staff engineering pass

Phase 17 audited correctness. This one asks a different question: is the thing
*engineered* well enough to publish as a reference implementation? Latency,
memory, concurrency, robustness to inputs the training data never contained,
and the two loose ends Phase 17 left open.

Three findings. Two are the loose ends; one is new and is the largest single
defect any pass has found in the serving path.

## 1 — the prediction endpoint spent 97% of its time rebuilding a cache

**Severity: High (performance).** `POST /api/v1/predict` is the product's
primary endpoint and the one the dashboard's what-if slider fires on every
drag.

Measured, before:

| endpoint | p50 |
|---|---|
| `GET /players/{id}` | 2.8 ms |
| `GET /players/{id}/history` | 8.9 ms |
| `GET /players/{id}/similar` (warm) | 10.0 ms |
| `GET /players?q=` | 14.2 ms |
| **`POST /predict`** | **385.3 ms** |

Forty times slower than anything else. The breakdown said why:

```
_preprocess (ColumnTransformer.transform)     4.8 ms
explainer(features)  -- the actual SHAP call  5.6 ms
pipeline.predict                              5.9 ms
building the TreeExplainer                  ~370   ms
```

`shap.TreeExplainer` walks the entire booster — 300 trees at 63 leaves — and
`explain_prediction` constructed a new one on **every request**, for an
estimator that never changes after the artifact is loaded. Inference itself was
1.6% of the endpoint.

**Fix.** Memoise the explainer per fitted estimator in
`src/explainability/shap_explainer.py`.

**After: 385.3 ms → 29.8 ms p50, 12.9x.** Output verified bit-identical across
25 random rows — max absolute SHAP difference 0.000e+00, max base-value
difference EUR 0.000e+00.

**The first attempt at this fix bought nothing, and the reason is worth
recording.** It used a `WeakValueDictionary`. Nobody else holds a reference to
an explainer, so every entry was collected between the call that created it and
the next one; the cache was structurally incapable of ever hitting, and the
endpoint stayed at 385 ms. The re-measurement caught it. The estimator is the
object with a natural lifetime, so it is the *key* — `WeakKeyDictionary`, strong
value — and a redeployed artifact still releases its explainer, which a plain
dict would not do. `tests/unit/test_explainability.py` asserts the cache hits,
that two artifacts never share an explainer, that the values are unchanged, and
that the entry is released when the model is.

## 2 — the search optimised an objective the project does not deploy

**Severity: Medium.** Phase 17 found this and did not measure it; the
measurement was killed for CPU. It is fixed rather than measured, because the
fix is unambiguously correct regardless of its size.

`src/models/tuning.py:_score_candidate` fitted each fold with
`pipeline.fit(train[features], train[target_column])` — **unweighted** — while
`src/pipelines/tune.py` fits the winner with `**fit_params(train["season"])`,
the recency weighting selected in `src/models/weighting.py`. So the grid was
ranked under one objective and the shipped model trained under another.

That is not leakage and it is not necessarily costly. It is simply wrong: a
hyperparameter chosen for an unweighted fit is not the one a weighted fit
wants. Both now fit with the same weights, and the models were retrained so the
artifacts match the code that would produce them.

**Measured, which is what Phase 17 could not finish.** The correction changes
nothing for the scouting model and changes the prior-value model's chosen
learning rate:

| variant | before | after |
|---|---|---|
| performance_only | `lr 0.1, leaves 63, mcs 120, λ 0` | **identical** |
| with_prior_value | `lr 0.1, leaves 63, mcs 20, λ 20` | `lr 0.05, …` |

and the prior-value test MAE moves the *wrong* way: EUR 1,624,273 to
EUR 1,637,639, R² 0.9174 to 0.9141.

That is the correct outcome and it is worth being explicit about why, because
the temptation is to revert. The old configuration was chosen by ranking a grid
under an objective the project does not deploy; it happened to land better on
the test seasons. Keeping it because the test number is prettier is selecting
on the test set through the back door — the precise failure this repository
spends most of its effort avoiding. The procedure is now right and the number
is what the right procedure produces. Every downstream claim was corrected
rather than quietly left: LightGBM is no longer the outright best family on
that variant, and four documents and one docstring said it was.

## 3 — no test asserted that anything came back

**Severity: Medium (process).** The gap that let Phase 17's two high-severity
bugs survive four previous audits.

Both `similar_players` and `prediction_history` returned `[]` — a 200, a valid
shape, and a dashboard panel reading "No comparable seasons found". Every test
asserted the *shape* of a response; none asserted it was populated. So the
suite stayed green while comparables were dead for 32.8% of the panel and 59%
of players had an empty career chart.

`tests/integration/test_api_live.py::TestNothingImportantIsSilentlyEmpty` now
runs against the real artifacts and asserts, for a player whose latest row is
the unlabelled current season — the exact population both bugs excluded — that
comparables come back, that a career history comes back, that the history
covers *every* labelled season on record rather than merely some, and that the
number of servable players equals the number of labelled players.

The last one is a population assertion rather than a spot check, which is the
point: a regression that empties an endpoint for a third of users passes every
single-player test ever written.

## What was attacked and held

**Robustness to unseen inputs.** The current season contains 5 citizenships, 1
competition type and 2 confederations the encoder never saw. All 402 affected
rows predict normally (`handle_unknown="infrequent_if_exist"`), as do invented
categories pushed through the API — `country_of_citizenship="Wakanda"`,
`position="Sweeper"`, `foot="both-and-neither"`. Unseen *clubs* and *leagues*
cannot break inference at all: `primary_club_id` and `primary_competition_id`
are not features, so a new league reaches the model only through continuous
strength features.

**Distribution drift.** Median KS between train and test across 48 numeric
features is 0.035. The largest are the censored career features
(`years_since_debut` 0.185, `seasons_observed` 0.151), which grow mechanically
with the panel and are already capped for that reason. Target KS is 0.095 —
median EUR 1.0M to 1.5M, football's own inflation, which is why `season` is not
a feature.

**Concurrency.** 160 concurrent `/similar` and 160 concurrent `/predict`
requests across 16 threads: all 200, no corruption, and the shared caches stay
bounded at 2 entries (variant x season, and variant).

**Abuse.** 5,000 unknown feature keys, a 1 MB body, a 60-deep nested array, a
negative limit — all 422, all rejected before any model work.

**Production shape.** Startup 1.27 s including two models, 86k rows and 50k
names. RSS 660 MB steady. Security: `yaml.safe_load` throughout; the one
`joblib.load` is documented with its threat model, mounted read-only, and
reachable by no endpoint.
