# Phase 19 — Codex audit remediation: ML correctness (F1–F4, F6, F7)

An independent, read-only Codex audit reported 20 findings. This pass fixes
the six that make features, labels or predictions wrong: F1–F4, F6 and F7.
Each was reproduced on the shipped tables and against the unfixed code before
anything changed. Every number below comes from a command against the same
raw snapshot (Kaggle files ingested 2026-08-31; the newest valuation is dated
2026-06-12).

The audit report itself was not in the repository. This pass worked from the
finding titles and reproduced each defect independently. F5 and F8–F20 are
**not** addressed here.

## Evidence preserved

The superseded tables, artifacts and reports were copied before the rebuild
to `archive/pre-remediation-2026-10-18/` (local, git-ignored, like `models/`
and `data/`). Checksums:

```
483dd14fba044ccf51c9dd146daeff8cb1695e4ce40fa4e7cfefb1e546ab3fd5  models/performance_only__lightgbm.joblib
e8263b4d5220bd508516725dd0abc1888f6aa54a9c4558a3e60389da052b2322  models/performance_only__lightgbm.json
8984e32183e9fcc0534fed5b6fee2576cc5957adce4b59c23d48d5b11091158a  models/with_prior_value__lightgbm.joblib
eff9ac05eedebba6a5f1d2e1c82fffa57d89f7189158ef0d20535c05a5a80b91  models/with_prior_value__lightgbm.json
341428ba1684e1a87aecca64eca18035869ffca7e086aa154dd7643aa74cab33  data/current_season.parquet
1be999c8dafbf45d65c56695b1c8acaad25a64588f0756da65f7f507dfa845bd  data/training_table.parquet
```

The previous model cards and `docs/model_comparison.md` are in git history
at `77ed342`, and copies sit in the archive.

## The common cause

Each defect respected the **season** and broke the **date**. A feature is
legal only if every input to it was available at the row's `as_of_date`.
"Same season" or "earlier season" does not guarantee that. The fix is
structural: every windowed feature now leaves a provenance date on the row,
and the leakage stage checks each one against `as_of_date` by value
(`Provenance` and `check_provenance_within_cutoff` in
`src/validation/leakage.py`, declared once as `FEATURE_PROVENANCE` in
`src/feature_engineering/build.py`). Event inputs (matches, lineups) may fall
on the cutoff. Published labels must fall strictly before it. A feature value
with no provenance date is itself an error, so dropping a date column cannot
switch the check off.

## F1 — future lineups in historical aggregates

**Defect.** `squad_role` counted every lineup in the season. A lineup is not
an appearance: an unused substitute is listed for matches they never played,
so lineups run past the last appearance, and therefore past `as_of_date`.
`starts`, `substitute_appearances`, `start_share`, `captain_share`,
`positions_played` and the lagged `prev_start_share` all carried it.
`check_no_future_dates` compares with the *label*, and nothing compared with
the as-of date.

**Reproduction.** On the shipped table, 5,816 training rows (6.77%) counted
lineups dated after their as-of date, and 307 of those counted lineups dated
after the label. The current-season table had 58. On the unfixed code, a
bench spot on 20 July counted toward a season closed on 1 July.

**Fix.** `squad_role(lineups, as_of)` keeps lineups with `date <= as_of_date`
per player-season and records `lineup_record_date`.

**Tests.** `test_performance.py`: `test_a_lineup_after_the_as_of_date_is_not_counted`,
`test_the_cutoff_is_per_player_season`. `test_leakage.py`:
`TestProvenanceWithinCutoff`, and `TestTheShippedContract` with a lineup
pushed one day past the cutoff.

**Rows affected.** Training rows using post-as-of lineups: 5,816 → 0. Current
rows: 58 → 0. Values changed: `starts` 1,469 rows, `substitute_appearances`
4,875, `start_share` 5,136, `captain_share` 783, `positions_played` 512,
`prev_start_share` 3,329.

## F2 — future-published labels in competition strength

**Defect.** `competition_value_level` is a target encoding. It used every
label from every *earlier season*, but with a 365-day label window, season
*s−1* can be labelled after season *s* has closed. The tier rank was taken
from it.

**Reproduction.** 8,413 training rows (9.79%) encoded at least one label
published on or after their own as-of date. That is 23,404 of 197.5M encoded
(row, label) pairs, concentrated in 2013–2014 and 2017 (2,065, 2,587 and 1,278
rows). On the unfixed code, a batch of labels published after the cutoff
moved the level (14.44 against 13.82 from public labels only).

**Fix.** `competition_strength(..., cutoffs=)` computes each season's level
from labels of earlier seasons published strictly before that season's
boundary. The boundary is never later than any as-of date in the season,
because `as_of = max(boundary, last appearance)`. The newest label used is
kept as `competition_level_source_date`, and the validator requires it to be
strictly before `as_of_date`.

**Tests.** `test_context.py`: `test_a_label_published_after_the_cutoff_is_not_used`
(an exact expected value), `test_the_source_date_is_strictly_before_every_cutoff`.
`test_leakage.py`: `TestTheShippedContract`.

**Rows affected.** Rows encoding a late label: 8,413 → 0.
`competition_value_level` changed on 27,224 rows,
`delta_competition_value_level` on 27,269, and `career_best_competition_level`
on 20,908. The null rate is unchanged at 7.11%.

**A definitional change to know about.** `competition_tier_rank` now ranks
among every competition with a level at the cutoff. Before, it ranked among
the competitions that had labelled rows *in season s*. That population
depends on season-*s* labels having been published, so it was not knowable
at the as-of date, and it did not exist at all for the current season (F3).
The rank changed on 57,771 rows.

## F3 — no competition strength on any current-season row

**Defect.** Levels were produced only for seasons present in the labelled
frame. The season being played has no labels, so all of its rows joined to
null.

**Reproduction.** `competition_value_level`, `competition_tier_rank` and
`delta_competition_value_level` were null on 8,709 of 8,709 current rows.
The imputer filled them with the training median, so every current player
was priced as if in a median-strength league.

**Fix.** The same as F2. Levels are computed for the seasons of the frame
being enriched, from history, so a season needs no label of its own.

**Tests.** `test_context.py::test_a_season_with_no_labels_of_its_own_still_gets_a_level`.

**Rows affected.** Current rows with a null level: 8,709 / 8,709 → 492 /
9,716 (5.1%, competitions with no qualifying history). The training null rate
is 7.1%. `delta_competition_value_level` is still null on 2,912 current rows
where the previous season had no level.

## F4 — two definitions of the prior value

**Defect.** Training took the *previous labelled season's label*
(`add_career_history`). Serving took the *latest raw valuation before the
as-of date* (`build_current_season_table`). These are different quantities:
the first skips every revaluation between that label and the as-of date, and
is missing for every first covered season.

**Reproduction.** Median `prev_value_age_days` was 242 in training against 34
in serving, and coverage was 71.6% against 93.4%. On the unfixed code, the
same player-season rebuilt through both builders got a 197-day prior from
one and a 32-day prior from the other.

**Fix.** One function, `attach_prior_value`: the latest valuation published
strictly before `as_of_date`, used by both builds. It records
`prev_value_date`, which the validator requires to be strictly before the
cutoff.

**Tests.** `test_feature_engineering.py::TestPriorValue`, including
`test_the_training_and_serving_builds_share_the_definition` (a labelled
season rebuilt through the serving path must come out identical) and
`test_a_revaluation_after_the_previous_label_is_used`.

**Rows affected.** `prev_market_value_in_eur` changed on 66,110 training rows,
and `prev_value_age_days` on 78,986. Training rows with a prior: 61,522 →
84,626 (98.4%). The training median age is now 33 days, against 34 in
serving. **The with-prior-value variant is therefore a different population
and a different, easier question**: it now anchors on a valuation typically a
month old rather than eight. That is the question serving always asked, but
its metrics are not comparable one-for-one with the old ones; see the
matched-row comparison below.

## F6 — current rows chosen by a global season cutoff

**Defect.** "Current" meant `season > max(training season)`. Once *any*
player's 2024 season was revalued, 2024 counted as labelled for everyone, and
every 2024 row still awaiting its valuation fell out of both tables.

**Reproduction.** 1,007 player-seasons from 2024 were in neither table. For
822 players this was the latest season. 457 of them had no servable row at
all, and the other 365 were served from older labelled seasons, one as far
back as 2011.

**Fix.** A player-season is current when it has no label in the training
table **and its label window is still open**: `as_of + tolerance` is later
than the newest valuation in the data. A row whose window closed without a
label is a gap in the source, not the present. The tolerance is now passed
through from the config.

**Tests.** `test_current_season.py::TestPlayerLevelLabelAvailability`, three
tests, plus `test_it_is_empty_once_every_open_window_has_closed` and
`test_it_holds_only_player_seasons_with_no_label`.

**Rows affected.** Current rows: 8,709 (all 2025) → 9,716 (8,709 from 2025
and 1,007 from 2024). The training and current tables share 0 keys.

## F7 — prior-value what-if without the prior-value inputs

**Defect.** `GET /players/{id}` always returned the *default* variant's
feature set. That is performance-only, which has no
`prev_log_market_value_in_eur` or `prev_value_age_days`. The what-if panel
sends that map to whichever variant is selected, so every prior-value
what-if went out without its prior. The fitted imputer filled it with the
training median and the API answered 200. The stored-row path had the same
hole: a NaN prior passed an `is None` check.

**Reproduction.** On the unfixed code, five of the new service tests fail:
the omitted prior, an explicit null prior and a stored NaN prior were all
answered, and the player payload ignored the variant.

**Fix.** `PredictionService._require_prior_value`, called from `_predict`,
which both prediction paths go through. It raises
`InvalidFeaturesError` (422) naming the missing inputs. `player()` and
`GET /players/{id}` take `variant`, and the dashboard passes the selected one.

**Tests.** `test_prediction_service.py::TestPriorValueInputs` (5),
`test_api.py::TestPlayers` (2).

**Rows affected.** 978 of 9,716 current rows (10.1%) have no prior
valuation: 578 from 2025, which were silently imputed before, and 400 of the
1,007 newly added 2024 rows. The prior-value variant now refuses them
instead of imputing, and the performance-only variant still prices them.

## Retraining and metrics

Both variants were retrained with `scripts/train_models.py`: the same eleven
families, the same grids, split, seed and selection rule. No fix, feature or
hyperparameter was chosen by looking at a test number. The test seasons were
scored once, by the pipeline, after selection. The 80% interval's coverage is
measured on the test seasons against quantiles from the validation season, as
before.

### Headline (each model on its own test rows)

| Variant | | Winner | Val MAE | Test MAE | Test R² | 80% coverage | Median width |
|---|---|---|---|---|---|---|---|
| performance-only | before | LightGBM | €1,938,817 | €2,068,081 | 0.8407 | 76.9% | €2,052,897 |
| performance-only | after | LightGBM | €1,908,448 | €2,079,666 | 0.8362 | 75.1% | €1,985,856 |
| with prior value | before | LightGBM | €1,656,909 | €1,637,639 (n 10,169) | 0.9141 | 79.3% | €1,862,759 |
| with prior value | after, first retrain (superseded) | Random Forest | €912,164 | €930,585 (n 13,315) | 0.9625 | 81.1% | €1,022,145 |
| with prior value | after, serving requirements | LightGBM | €900,492 | €912,394 (n 13,315) | 0.9629 | 80.3% | €1,022,809 |

**Performance-only: no gain from removing the leaks, and slightly worse on
test.** It uses the same 13,486 test rows, so the comparison is direct. Test
MAE is €11,585 worse (+0.56%, well inside one standard error of €39,786),
R² is 0.0045 lower, and interval coverage fell 1.7 points further below
nominal. Per season: 2023 is €2,196,921 → €2,200,626 and 2024 is
€1,973,011 → €1,990,411. Validation MAE improved by €30,369. The leaked
information was worth roughly nothing to accuracy; removing it costs a
little, and the reported numbers now describe what the model can know.

**With prior value: much better, mostly because the question changed.** It
now anchors on the newest valuation before the as-of date, a median 33 days
old rather than 242, which makes the next valuation far easier to predict.
That is the question serving was already asking, so the old headline
described a model nobody was actually running.

### Controlled comparison (prior-value variant)

The row sets differ, so three controlled views (`old` = archived artifact
and features):

| Model | Features | Rows | Test MAE | R² | 80% coverage |
|---|---|---|---|---|---|
| old | old (training definition) | 10,169 | €1,637,639 | 0.9141 | 79.3% |
| old | **serving-built** (fresh prior) | same 10,169 | €1,307,633 | 0.9393 | **88.9%** |
| new | new | same 10,169 | €1,027,642 | 0.9649 | 84.1% |
| new | new | all 13,315 | €930,585 | 0.9625 | 81.1% |

The second row is what F4 cost in production. The old model was fitted on
stale priors and served fresh ones: its errors there were smaller than its
reported test figure, but its interval covered 88.9% where it claimed 80%.
The published headline and the published interval both described a
different input distribution from the one served. On the same rows, the
retrained model is €280k better than the old model fed serving-built rows.

### Selection changed for the prior-value variant (first retrain; superseded below)

LightGBM had the lowest validation MAE (€900,492). Four families sat within
one standard error (€27,090), and the existing tie-break, deployment
footprint in the serving image, picked Random Forest at 0 MB of extra
packages against LightGBM's 9 MB. The rule was not changed after the fact.
Its consequences, which the rule does not weigh:

* **Artifact size** is 182 MB, against 2.2 MB.
* **Prediction latency** for `POST /predict` with an explanation is
  **439 ms** p50, against 29 ms for the LightGBM variant, measured through
  `PredictionService`. That undoes much of Phase 18's latency work for this
  variant, and the what-if slider calls it on every debounced drag.
* **Reproducibility** is 1e-9 € rather than exact, because forest
  prediction parallelises (`n_jobs=-1`) and summation order varies.
* **The report build** took 13.5 minutes, mostly global SHAP over the forest.

This is a product decision for the owner. Shipping LightGBM for this variant
would need a change to the selection rule (for example, counting artifact
size or latency in the footprint), made on its own merits and before looking
at any test number.

## Selection policy: serving requirements

**Why the one-standard-error rule picked Random Forest.** The rule was
correct as written but measured the wrong cost. Its footprint counted only
extra packages in the serving image, which is 0 MB for every scikit-learn
family and 9 MB for LightGBM. It ignored the artifact itself (182 MB against
2.2 MB) and the cost of serving an explanation (437 ms against 24 ms). Within
the band, a heavier and slower model therefore looked cheaper.

**The policy** (`src/pipelines/tune._select_winner`, committed in `16def0b`
and `f935b23` before the retrain that applied it). It names no family and
reads only training and validation-season measurements, all recorded per
family in the artifact's leaderboard:

1. **Eligible** only if the family exposes feature importances (the existing
   `EXPLAINABLE_FAMILIES`, unchanged) *and* explains a single prediction:
   SHAP builds an explainer and every timed validation row explains without
   error.
2. **Within budget.** The median of predict plus SHAP explanation over 25
   validation-season rows must be at most `SERVING_LATENCY_BUDGET_MS` = 100 ms.
   The budget comes from the product, not from these models: the what-if
   slider debounces at 300 ms, and Phase 18 rated a 385 ms response High
   severity.
3. **One standard error** around the best *eligible* validation MAE.
4. **Tie-break:** the smallest deployed footprint (package megabytes plus
   artifact megabytes), then latency, then validation MAE.

If no family is eligible, training stops with an error rather than shipping
something that fails the requirements. The same rule runs for both variants.

**Applied by the retrain (exit 0, 2026-10-18 16:49):**

| Variant | Family | Val MAE | SE | Explained p50 | Deployed | Outcome |
|---|---|---|---|---|---|---|
| with prior | LightGBM | €900,492 | €27,090 | 24.4 ms | 11.2 MB | **selected** |
| with prior | XGBoost | €906,357 | €27,444 | 35.3 ms | 373.4 MB | in band, larger |
| with prior | Extra Trees | €908,924 | €27,402 | 792 ms | 297.9 MB | over budget |
| with prior | Stacked | €911,068 | €28,286 | — | 701.3 MB | no importances |
| with prior | Random Forest | €912,164 | €26,998 | 437 ms | 182.0 MB | over budget |
| with prior | HistGradientBoosting | €913,440 | €29,127 | 29.8 ms | 0.5 MB | no importances |
| with prior | CatBoost | €931,292 | €28,307 | — | 328.2 MB | cannot explain |
| perf-only | LightGBM | €1,908,448 | €54,344 | 25.7 ms | 11.2 MB | **selected** |
| perf-only | Stacked | €1,932,039 | €54,091 | — | 701.3 MB | no importances |
| perf-only | XGBoost | €1,956,135 | €56,764 | 33.9 ms | 373.4 MB | in band, larger |
| perf-only | Random Forest | €2,276,732 | €70,978 | 3,939 ms | 894.5 MB | over budget |
| perf-only | Extra Trees | €2,277,378 | €71,934 | — | 1,592.8 MB | additivity failure |

Linear families fail eligibility on both variants: they expose coefficients,
but TreeExplainer cannot explain them.

HistGradientBoosting is the one family to look at closely. It would win the
tie-break at 0.5 MB, and SHAP works on it. It is excluded by the
`UNEXPLAINABLE_FAMILIES` entry from 2026-08-31 (`6355369`), because it has
no `feature_importances_` and the model card and `/model` table would come
back empty. That exclusion predates this work and was not revisited here.

**Test metrics, read once after selection was frozen** (the policy chose
nothing from them):

| Variant | Model | Rows | Test MAE | R² | 80% coverage |
|---|---|---|---|---|---|
| with prior | LightGBM (new) | 13,315 | €912,394 ± 19,452 | 0.9629 | 80.3% |
| with prior | Random Forest (first retrain, archived) | 13,315 | €930,585 | 0.9625 | 81.1% |
| with prior | LightGBM (new), controlled rows | 10,169 | €1,003,418 | 0.9651 | 83.7% |
| perf-only | LightGBM (new) | 13,486 | €2,079,666 ± 39,786 | 0.8362 | 75.1% |

The performance-only model is unchanged from the first retrain (same family,
same parameters, same metrics). The €18k test gain over Random Forest is
inside one standard error and is not a claim of better accuracy. The reason
for the switch is serving cost: the artifact is 2.2 MB instead of 182 MB, an
explained prediction takes 24 ms instead of 437 ms, and the prediction is
exactly reproducible.

**Checks added.** In `tests/unit/test_model_selection.py`,
`TestServingRequirements` covers that the budget is inclusive, that the band
is anchored on the best eligible family, that a model with no SHAP is
excluded, that the footprint counts the artifact, that no eligible family
is an error, and that both variants get the same answer. In
`tests/integration/test_serving_performance.py`, the tests run against every
shipped artifact: it explains a prediction, it meets the budget on
validation rows, the choice can be recomputed from its own leaderboard, and
there is one artifact per variant. A scratch check explained every labelled
and current-season row with both shipped models: 189,046 rows, with no
additivity failure and a largest reconstruction error of 2e-12 in log space.

## Validation

* Every new regression test was also run against the unfixed `HEAD` in a
  scratch worktree: the F6 and F7 tests fail there. The F1, F2 and F4 tests
  import the new APIs, so equivalent probes were run on the old functions;
  each showed the defect.
* `make test-cov`: 713 tests, 90% coverage. The only failures are three
  `test_authorship_guard` cases, which fail identically before this change:
  the test pins `PATH=/usr/bin:…`, and `/usr/bin/git` on this machine is the
  Xcode shim, which refuses to run until the Xcode licence is accepted.
* Leakage stage on the rebuilt tables: 0 errors, 0 warnings, with the
  provenance checks active on both the training and the current-season table.
* ruff, black and mypy are clean. Frontend `tsc`, `eslint` and 96 vitest
  tests pass.

## Remaining risks

* **F5 and F8–F20 are untouched.** The full audit report was not available.
* **Pending rows may never be labelled.** A 2024 row stays "current" until
  its 365-day window closes (to 2026-07-31). Some of the 1,007 belong to
  players who have left Transfermarkt's active coverage. They are priced as
  current until then.
* **In-progress seasons have a future as-of date.** A season still being
  played has `as_of_date` at its coming boundary, so age and career length are
  measured at that date. That is unchanged by this pass, and harmless on this
  snapshot (the boundary, 2026-07-01, has passed), but a mid-season snapshot
  would describe players slightly older than they are.
* **Tier rank population.** `competition_tier_rank` now ranks among
  competitions with a level at the cutoff (F2). A competition that dropped out
  of coverage keeps a stale level in that ranking.
* **Momentum features have no provenance date.** The `prev_*` performance
  lags rely on as-of dates being monotone within a player. A test asserts
  that, but it is not a per-row value check.
* **Docs not re-quoted.** `docs/demo_script.md` and the narrative sections of
  `docs/METHODOLOGY.md` still quote pre-remediation figures as history. The
  README headline, zoo table and test counts, both model cards and
  `docs/model_comparison.md` are current.
