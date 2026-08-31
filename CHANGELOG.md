# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
