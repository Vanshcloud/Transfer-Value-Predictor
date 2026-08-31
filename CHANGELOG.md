# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] — 2026-09-01

A research-grade audit that re-verified every claim from the raw files rather
than from the previous phase's notes. It found one real leak, one overstated
metric, one feature block worth adding — and rejected six plausible ideas on
measurement. Full workings in
[`plans/06-final-research-audit.md`](plans/06-final-research-audit.md).

### Fixed

- **`POST /api/v1/predict` was 385ms; it is now 30ms.** `shap.TreeExplainer`
  walks the whole booster (300 trees at 63 leaves) and was being rebuilt on
  every request for an estimator that never changes. **97% of the endpoint's
  latency was that rebuild** — the actual SHAP call is 5.6ms and inference
  5.9ms. Now memoised per fitted estimator with a `WeakKeyDictionary`, so a
  redeployed artifact still releases its explainer. Output verified
  bit-identical over 25 rows: max SHAP difference 0.000e+00.
- **The hyperparameter search fitted unweighted while the final model fitted
  weighted.** `src/models/tuning.py:_score_candidate` ranked the grid under an
  objective the project does not deploy. Both now use the same recency
  weighting, and the models were retrained so the artifacts match the code.
- **`similar_players` returned nothing for every currently-playing player.**
  It anchored on the player's most recent row, which since v1.2.0 is the
  unlabelled season being played; the neighbour pool is labelled-only, so the
  pool was empty and the endpoint returned `200` with `[]`. **8,709 players,
  32.8% of the panel** — and the dashboard rendered "No comparable seasons
  found", so it looked like an answer. Now anchors on the latest *labelled*
  season: 144 of 200 sampled active players get comparables, against 0 before.
- **65.3% of rows were hidden from the career chart and the comparables pool.**
  `_rows_for_variant` and `prediction_history` required all 54 features to be
  non-null, in a pipeline that carries a fitted imputer. That dropped 56,148 of
  85,966 labelled rows including **every player's first season**, and left
  **14,402 of 24,411 players (59%) with an empty history**. The gate is now the
  target — and the prior value for the variant defined by it — and nothing
  else. Pre-existing (the 41-feature list dropped 56.5%), worsened by the
  momentum block, fixed now.
- **Implausible feature values were answered rather than rejected.**
  `data.min_age: 15` and `data.max_age: 45` were declared in
  `configs/config.yaml`, parsed and cross-validated in `src/utils/config.py`,
  and read by nothing — so `POST /api/v1/predict {"features": {"age": 1e12}}`
  returned `200` and a confident number. Now enforced via `PLAUSIBLE_RANGES`,
  verified against 0 violations in 85,966 training and 8,709 serving rows.
- **The leakage validator never saw 28.4% of the table, or the serving table
  at all.** It checked only the prior-value variant — every column, but 61,522
  of 85,966 rows. Both now validated, plus `current_season`, which is what the
  API answers from. All three clean: 0 errors, 0 warnings.
- **Five stale numeric claims in docstrings**, all rationale rather than
  behaviour: target skew 8.70 → 6.16, prior-value skew 4.53 → 5.53, the split
  R² figures in `src/models/splits.py`, and the cost of the explainability
  constraint. Every conclusion survives; one flipped sign in the project's
  favour.
- **Label leakage in the club-form features.** `club_points_per_game`,
  `club_goal_difference_per_game`, `club_league_position`, `club_matches` and
  `squad_match_share` were whole-season means. A season runs August–July, so a
  fixture on 20 July belongs to it while falling after the 1 July as-of
  boundary — **5.56% of all fixtures are dated in July**. The mean therefore
  included matches played after the row's as-of date on 17.6% of rows, and
  after the **label** on 1,049 of them (1.22%). Seven leakage checks missed it
  because it lived inside a number, not a column name or a date.
  `club_season_strength` now returns a running record joined with `merge_asof`
  on the row's own date. Removing it cost nothing measurable (p = 0.49, 0.89).
- **The prediction interval was calibrated on the test set it was scored
  against**, which makes a nominal 80% interval cover exactly 80% by
  construction. Quantiles now come from the validation season and coverage is
  measured on the test seasons: **0.763 achieved against 0.80 nominal** for the
  scouting model. Served as `confidence.measured_coverage`, printed in both
  model cards, and shown in the dashboard beside the nominal level.

### Added

- **13 career-momentum features** — last season's minutes, appearances,
  contributions, start share, competition level, club form and squad share; the
  season gap; season-over-season deltas; expanding career means and maxima.
  Lagged *features*, never a lagged label, so they are legal in the variant that
  withholds the prior valuation. Pooled over five held-out seasons at three
  seeds: **performance-only €1,976,268 → €1,850,058 (−6.4%, t = 11.30)**,
  improved in 5/5 seasons.

### Headline metrics

Test seasons, which the model and the interval have both never seen.

| variant | v1.2.0 | now | |
|---|---|---|---|
| performance-only | €2,205,618 / R² 0.813 | **€2,068,081 / R² 0.841** | −6.2% MAE |
| with prior value | €1,661,311 / R² 0.914 | **€1,637,639 / R² 0.914** | −1.4% MAE |

The prior-value figures moved slightly *worse* between the audit and release —
€1,624,273 to €1,637,639 — because the hyperparameter search was corrected to
fit under the recency weighting the final model actually uses. The earlier
number came from a configuration chosen under an objective the project does not
deploy; it happened to land better on the test seasons, which is exactly the
kind of thing not to chase. Among the explainable families LightGBM is within
one standard error of the best and wins the deployment-cost tiebreak.
- **`min_child_samples` and `reg_lambda` in the LightGBM grid.** Worth €19,846
  (p = 0.0087) and €22,214 (p = 0.0014) of pooled held-out MAE, and the reason
  knowledge distillation was rejected — see below.
- **`measured_coverage`** on the prediction response, and `career_stage` plus
  `primary_competition_id` segments in the error report.
- `check_no_future_dates` now runs over the whole frame rather than the feature
  subset, and the as-of club join keeps its source date as `club_record_date`.
  A numeric aggregate cannot be audited directly; the provenance timestamp of
  the join that built it can.

### Added

- `tests/integration/test_api_live.py::TestNothingImportantIsSilentlyEmpty` —
  the guard for the class of bug that produced two high-severity findings: a
  `200` with an empty body. It asserts, against the real artifacts and for a
  player whose latest row is the unlabelled current season, that comparables
  and history come back, that history covers *every* labelled season rather
  than some, and that servable players equal labelled players. A population
  assertion, because a regression that empties an endpoint for a third of users
  passes every single-player test.

### Repository

- `SECURITY.md`, `CODE_OF_CONDUCT.md`, `ROADMAP.md`, `PROJECT_STRUCTURE.md` and
  `docs/demo_script.md` are new. `ROADMAP.md` lists only work a measurement
  says is open; ideas that were built and rejected live in `plans/` and are
  deliberately not restated as future work.
- Model cards gain **Assumptions**, **Failure modes**, **Fairness** and
  **Ethical considerations**. Fairness is a measured table rather than a claim —
  MAPE by competition, which spans 36% to 102%. Written in the generator, not
  the output, so the next report build cannot undo it.
- README gains an architecture diagram, a Testing section, citations,
  acknowledgements and future work. Two stale counts fixed (53 skips, not 49;
  54 features, not 41) and a generated model-card figure corrected — "roughly
  60% worse than a random split" was measured at 45% against a grouped one.
- `pyproject.toml` gains `[project.urls]` and classifiers; `.env.example` now
  documents `CORS_ORIGINS`, which the API reads and compose sets but nothing
  told a deployer about. Empty `src/preprocessing/` package removed.

### Measured and rejected

Every one of these was implemented, measured, and thrown away.

- **Knowledge distillation.** A LightGBM student taught by the stacked blend
  beat the shipped model by €22,333 (p = 0.0006) while staying explainable.
  Against a *regularised* LightGBM it is worth €2,487 (p = 0.72); a LightGBM
  teaching itself gains nothing (p = 0.81). The ensemble was substituting for
  regularisation the grid had never been allowed to try.
- **Indexing seasons by each competition's own calendar.** First looked like a
  10% win; it was an artefact of the as-of date moving earlier on 93% of rows
  and cutting the median label horizon from **141 days to 23**. With the
  horizon held fixed: null for the scouting model, **significantly worse**
  (−€68,996, p = 0.0005) with prior value, on 6.5% fewer rows.
- **Pre-2012 rows.** 40,339 of them *can* be reconstructed from valuations and
  biography. 34 of 54 features are structurally missing; adding all of them
  moved MAE by 0.19%, **p = 0.73**.
- **Club home attendance.** −€297,445, p = 1e−65. Season 2020 was played behind
  closed doors: mean attendance 14,769 → 2,752, correlation with value
  +0.65 → +0.24. The column measures a pandemic for one year in fourteen.
- **Transfer history** (p = 0.74, 0.35) and **club strength trajectory**
  (−€35,695, p = 0.023 — actively harmful).
- **Three alternative targets.** Delta-value (p = 0.37), within-season
  percentile (identical Spearman, 50–66% worse in EUR because the inverse map
  needs a distribution nobody has at prediction time), career peak
  (right-censored on 28.4% of rows, non-randomly).
- **Conformalised quantile regression, split conformal, and raw LightGBM
  quantile regression.** All lose on Winkler score to the band-wise method
  already here, once that one is calibrated honestly.

## [1.2.0] — 2026-08-31

Six of the seven "unavoidable dataset limitations" from the v1.1.0 audit turned
out to be limitations of the pipeline. `configs/config.yaml` downloaded three of
the ten files this Kaggle dataset ships; the other seven carried competition
identity, club results, starting line-ups and transfer fees.

### Changed

- **Label window 120 days → 365.** The old window closed on 29 October and
  missed Transfermarkt's winter revaluation batch, discarding 61% of the panel.
  Measured: 39.0% of player-seasons labelled at 120 days, 74.7% at 180, 90.8% at
  365. Still a strictly forward join, so it cannot leak.
  **36,880 rows → 85,966; 17,053 players → 24,411.**
- **19 features → 41.** Squad role and captaincy from `game_lineups.csv`,
  availability against the club's own fixture count, consistency and trajectory
  from the match rows, club results from `club_games.csv`, and competition
  identity and strength.
- **Test MAE €4.44M → €2.21M (R² 0.441 → 0.813)** for the scouting model, and
  **€3.71M → €1.66M (R² 0.775 → 0.914)** with a prior valuation.
- **Selection is no longer `min()`.** Two rules: the shipped model must produce
  named feature importances, and among families within one standard error of the
  best, the cheapest to deploy wins.

### Added

- `competition_value_level`, a league-strength feature computed on a strictly
  expanding window so a season never contributes to its own value.
- **Current-season predictions.** 8,709 rows for the season being played, built
  through the same enrichment code as the training table.
- **A seventh leakage check.** A year-wide label window lets season *s* be
  labelled after *s+1* begins, making *s+1*'s "previous value" not previous —
  22 rows in 61,555.
- Extra Trees and a stacked ensemble (11 families), `Metrics.mae_standard_error`,
  recency-based season weighting, and an optional `transfer_fee` target.
- Sample-data slices of the four context tables, so CI runs the new code rather
  than only type-checking it.

### Not fixed

- **Coverage still begins in 2012.** Before 2012-07-03 this dataset holds 2,470
  events across 190 games — international tournaments — and no line-up data
  before 2013. It needs a different source, not better use of this one.

## [1.1.0] — 2026-08-30

A full external release audit, and the remediation of everything it found. No
new capability; the change is that far more of what this repository claims is
now checked by something other than a person reading it.

### Fixed

- **`POST /api/v1/predict` returned a bare `500` on a malformed feature value.**
  Feature *names* were validated and feature *values* were not, so a wrong type
  reached the fitted pipeline and scikit-learn raised from inside a
  transformer. The response was `text/plain` "Internal Server Error" — for
  input a caller typed — while `api/errors.py` and `docs/API_CONTRACT.md` both
  claimed every non-2xx used one envelope. Values are now checked at the
  boundary (non-scalars, non-numbers, booleans, non-finite numbers, negatives
  for features that cannot be negative, absurdly long categories), every
  offending value is reported at once with its feature named, and handlers for
  Starlette's `404`/`405` and for any unhandled exception make the envelope
  claim literally true.
- **Host-specific detail survived in the git history.** `plans/00-discovery.md`
  said local paths, an account name and a machine inventory "has been removed";
  they sat in 16 of 17 commits, and the removing commit rendered every line
  again in its own diff. History rewritten with `git filter-repo`, verified
  gone, and a CI grep over `git log -p --all` prevents recurrence.
- **The documented coverage figure was not the one the documented command
  printed.** README said 97% and cited `make test-cov`, which prints 89%; 97%
  needs the data-dependent suite. Both are now stated beside the command that
  produces each, both are enforced with `--cov-fail-under`, and CI runs the
  first on every push.
- **CI's clean-checkout guard proved nothing.** `test ! -d
  data/processed/players.parquet` tested for a *directory* against a file, so
  the step guarding this project's signature property passed unconditionally.
- **The README's endpoint table listed nine of eleven routes** — the same two
  Phase 13 had just added to the API contract.
- `eur(999_999)` rendered `€1000k` rather than `€1.00M`; the thresholds were
  unit boundaries rather than the point where the smaller unit stops fitting.
- `useAsync` refetched forever, silently, if its fetcher was not wrapped in
  `useCallback`. All three call sites were correct, so nothing was broken, but
  the next page added would have self-DoSed the API. It now says so, once, in
  development.
- `PredictionService`'s docstring claimed a broken artifact fails at startup;
  it is logged and skipped, which is the better behaviour.

### Added

- **A frontend test suite.** 77 tests over 2,804 lines that previously had no
  test runner installed: the error mapping in `lib/api`, the race guard in
  `useAsync`, money formatting, the states each component renders, and that a
  per-feature euro figure never appears beside a SHAP contribution.
- **A text alternative for every chart.** Plotly emits positioned shapes; the
  plot is now `aria-hidden` with a visually-hidden table of the same series
  derived from the same prop, so the two cannot diverge.
- **Locked dependencies.** `requirements-lock.txt` pins all 55 packages
  including transitives; `requirements.txt` stays the reasoned declaration.
  `tests/unit/test_dependencies.py` fails when they disagree.
- **A serving-only dependency set.** `requirements-serve-lock.txt` drops
  xgboost and catboost — 291 MB of unused CUDA libraries, 269 MB of catboost,
  59 MB of plotly. The API image went from **2.62 GB to 1.32 GB** with an
  identical prediction on the same input.
  `tests/integration/test_serving_dependencies.py` fails if a tuning run ever
  ships an artifact the serving image cannot unpickle.
- Tests that check the README's stated numbers against pytest's own collector,
  that check the endpoint tables in the README and the contract against the
  live OpenAPI document, and that check the release tag against the declared
  version — the drift that started all of this.
- CI steps for the frontend suite, coverage floors, every commit's author, and
  host-specific detail in the history.
- Dependabot with the ML stack grouped and pandas 3 held back deliberately;
  issue templates for the three reports CONTRIBUTING asks for; a HEALTHCHECK
  on the dashboard image.

### Removed

- `notebooks/` and `data/external/`, which held nothing but `.gitkeep`, and
  `Settings.external_dir`, which nothing read.
- `scripts/` from the API image: it needs packages that image no longer
  installs, so it was shipping commands that raise `ImportError`.

## [1.0.0] — 2026-08-30

First public release. Predicts the market value of professional footballers and
explains every prediction.

### Added

- **Ingestion** — Kaggle `davidcariboo/player-scores` (CC0) behind a source
  Protocol, over a polite retrying `requests` client. DuckDB + Parquet storage
  behind a storage Protocol, so PostgreSQL is a second implementation rather
  than a rewrite.
- **Validation** — table contract checks plus an explicit leakage stage:
  `LeakageValidator` bundles six checks and re-runs after every split.
- **Features** — one row per player-season, labelled by an as-of join with a
  120-day tolerance rather than an equality merge, because valuations are an
  irregular on-change event series.
- **Models** — three-way temporal train/validation/test split as the headline
  evaluation; nine tuned families searched on expanding-window folds; two
  shipped variants, `performance_only` and `with_prior_value`. Versioned joblib
  artifacts with a readable JSON sidecar and a schema-version guard on load.
- **Explainability** — SHAP contributions on every prediction, additive in log
  space, with an `effect_multiplier` giving the multiplicative reading.
- **Evaluation** — error analysis, model comparison, and a model card per
  variant, all regenerated by `scripts/build_reports.py`.
- **API** — FastAPI over a service layer that imports no web framework. Eleven
  endpoints, prediction intervals, an unversioned `/health` distinguishing
  liveness from readiness, and a written contract in `docs/API_CONTRACT.md`.
- **Dashboard** — Next.js 16 / React 19 / Tailwind v4 / Plotly. Search,
  prediction with interval and SHAP, similar seasons, season-by-season
  prediction history, percentile radar comparison, and an interactive what-if.
- **Operations** — multi-stage Dockerfiles running as a non-root user, compose
  for both services, and CI enforcing lint, types, tests, authorship, removed
  APIs and two architectural boundaries.

### Deliberately not included

- **Transfermarkt is never scraped.** Its Terms of Use §11.1 prohibit automated
  access and separately prohibit training a model on the content. Labels come
  from the CC0 Kaggle mirror, and CI greps to keep it that way.
- **FBref enrichment.** Spiked and declined on measured evidence: reachable only
  via Selenium, which this project bans; the columns duplicate data already
  joined on a real integer `player_id`; and the documented fallback covers only
  a season that lands entirely inside the test split. See
  `plans/02-fbref-spike.md`.
- **PostgreSQL and Redis.** Both wait for a measurement that justifies them.

### Fixed in the release audit

- Two live endpoints — `/players/{player_id}/history` and
  `/features/distribution` — were missing from the API contract. Documented, and
  `tests/unit/test_api_contract_sync.py` now fails if the two ever diverge.
- MIT was declared in `pyproject.toml` with no `LICENSE` file.
- `beautifulsoup4`, `lxml` and `plotly` were installed and imported nowhere.
- `--help` ran the pipeline on three scripts instead of describing it.
- The version was declared `0.1.0` in five places. There is now one Python
  source of truth, and a test that fails when the survivors disagree.
- CI had no `permissions` block and no guard on the Transfermarkt rule.
- `/compare`'s two search inputs had no accessible label.

[1.2.0]: https://github.com/Vanshcloud/Transfer-Value-Predictor/releases/tag/v1.2.0
[1.1.0]: https://github.com/Vanshcloud/Transfer-Value-Predictor/releases/tag/v1.1.0
[1.0.0]: https://github.com/Vanshcloud/Transfer-Value-Predictor/releases/tag/v1.0.0
