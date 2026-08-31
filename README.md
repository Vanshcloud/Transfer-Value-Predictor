# Transfer Value Predictor

Predict the market value (EUR) of professional footballers from performance,
biographical and contextual data — and explain every prediction.

Search a player, get a valuation with a confidence interval, see the SHAP
contributions that produced it, compare him to the seasons nearest him in the
model's own feature space — then change the inputs and watch the number move.

Metrics are reported on **held-out seasons the model has never seen**: test MAE
**€2.21M** (R² 0.813) for the scouting model, **€1.66M** (R² 0.914) with a prior
valuation. A random split would read better and answer a question nobody
deploying this will ever ask.

**Try it without signing up for anything:** `make setup && make test` runs the
suite against the committed sample data — no Kaggle account, no download, no
trained model. Roughly fifteen seconds.

![A player's valuation, its 80% interval, and the SHAP contributions that produced it](docs/img/player-detail.png)

*Every prediction arrives with the interval it earned, the features that moved
it — read multiplicatively, because the model predicts log value — and the model
that produced it.*

## Why this repository looks the way it does

The architecture was decided *after* examining the data, not before. Three
findings shaped it, all recorded with evidence in
[`plans/00-discovery.md`](plans/00-discovery.md):

- **Transfermarkt is never scraped.** Its Terms of Use §11.1 prohibit automated
  access *and* separately prohibit using the content to train machine-learning
  models. Labels come from the CC0-licensed Kaggle mirror
  `davidcariboo/player-scores` instead.
- **There is no entity-resolution problem.** `appearances.csv` ships in the same
  CC0 download as the labels and shares its `player_id`, so features and labels
  join on an integer key. FBref was spiked as optional enrichment and declined —
  it is 403-Cloudflared to `requests`, reachable only via banned Selenium, and
  the columns it returns duplicate `appearances.csv` on a worse (name-based) key
  ([`plans/02-fbref-spike.md`](plans/02-fbref-spike.md)).
- **Temporal evaluation is the headline.** A measured spike
  ([`plans/01-feasibility-spike.md`](plans/01-feasibility-spike.md)) put EUR MAE
  at €2.37M under a group split and €3.96M under a temporal one. The temporal
  number is the one that reflects deployment, so it is the one reported.

## Setup

```bash
make setup
```

**macOS prerequisite.** LightGBM and XGBoost both link `@rpath/libomp.dylib` and
bundle no copy of it, so without this they fail at import with
`OSError: dlopen ... Library not loaded`:

```bash
brew install libomp
```

scikit-learn bundles its own OpenMP, which is why "sklearn works but LightGBM
doesn't" is such a common and confusing report. On Linux and in Docker the
manylinux wheels bundle libgomp and no action is needed.

## Pipeline

**First, data access.** `fetch_data.py` downloads the CC0 Kaggle mirror
`davidcariboo/player-scores`, which needs a free Kaggle account. Create a token
at [kaggle.com/settings](https://www.kaggle.com/settings), then either:

```bash
cp .env.example .env        # fill in KAGGLE_USERNAME and KAGGLE_KEY
```

or place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`. Without one of
those, `fetch_data.py` stops with a message naming both options rather than
failing obscurely.

**You do not need an account to run the project.** `data/sample/` is committed,
so `make test` gives you 657 passing tests and 49 skips with no credentials at
all — the 49 are the integration tests that need the full panel. `make test`
deselects them by marker; a bare `pytest` on that same clone *skips* them and
still reports zero failures, which is the stronger property and the one CI
checks. Only the three pipeline commands below need the
download.

Three stages, each reading what the last one wrote. Every figure below is
printed by the command above it, so nothing here is a number someone typed in.

```bash
python scripts/fetch_data.py       # Kaggle -> data/raw -> data/processed (parquet)
python scripts/validate_data.py    # contract checks; --strict fails on warnings too
python scripts/build_features.py   # one row per player-season, labelled and leak-checked
python scripts/train_baseline.py   # baselines across all three splits
python scripts/train_models.py     # nine tuned families, best saved per variant
python scripts/build_reports.py    # evaluation, SHAP, error analysis, model cards
```

The training table:

| | before Phase 15 | now |
|---|---|---|
| rows | 36,880 | **85,966** |
| players | 17,053 | **24,411** |
| seasons | 2011–2024 | 2011–2024 |
| rows with a prior-season value | 19,827 | **61,522** |
| features | 19 | **41** |
| current-season rows, predictable | 0 | **8,709** |
| leakage findings | 0 | 0 |

The label is the first valuation recorded **after** the season's evidence is
complete — an as-of join, never an equality merge, because valuations are an
irregular on-change event series.

That join used a **120-day** tolerance and discarded 61% of the panel. A season
ending 1 July gives a window closing 29 October, which is before Transfermarkt's
winter revaluation batch exists. Measured across the full panel:

| Tolerance | Player-seasons labelled |
|---|---|
| 120 days | 39.0% |
| 180 days | 74.7% |
| **365 days** | **90.8%** |

Widening cannot leak: `direction="forward"` is unchanged, so every label is
still strictly after the features. It did expose a leak — with a year-wide
window, season *s* can be labelled after season *s+1* has begun, making *s+1*'s
"previous value" not previous. 22 rows in 61,555, and now a seventh leakage
check ([`src/validation/leakage.py`](src/validation/leakage.py)) that fails the
build if it recurs.

Two model variants come out of the one table: *performance-only* (every row;
the useful model, for scouting) and *with prior value* (61,522 rows; the
accurate model, for tracking). Shipping only the second would be technically
true and practically useless.

## What the model sees

41 features, from every file the Kaggle dataset ships. The project previously
downloaded three of its ten, which is why "no league strength" and "five
performance statistics" were listed as limitations of the data rather than of
the pipeline.

| Group | Features |
|---|---|
| **Match output** | appearances, minutes, goals, assists, yellow/red cards, goal contributions |
| **Rates** | goals/90, assists/90, cards/90, contributions/90, minutes per appearance |
| **Squad role** | starts, substitute appearances, start share, captain share, positions played |
| **Availability** | share of the club's matches played, months active |
| **Consistency** | full-match share, minutes variability, scoring-match share |
| **Trajectory** | second-half goal share |
| **Club context** | points per game, goal difference per game, league position |
| **Competition** | value level, tier rank, competitions played, continental minutes share, type, confederation |
| **Biography** | age, age², height, position, sub-position, foot, citizenship |
| **Career stage** | years since debut, seasons observed |
| **Prior value** *(second variant only)* | lagged log value, its staleness in days |

Two of these needed care rather than code.

**`competition_value_level`** is the honest answer to "how strong is this
league", and the honest answer is the market value of the players in it — which
is the target. Computed from the current season that is textbook target
leakage. It uses a **strictly expanding window**: the level for a competition in
season *s* is the mean of seasons *< s* only, via `shift(1)` then
`expanding()`. Asserted three ways in
[`tests/unit/test_context.py`](tests/unit/test_context.py), including that
perturbing the last season cannot change any earlier feature.

**Club strength** comes from `club_games.csv` — actual results — and not from
`clubs.csv`, whose squad value and size are *current* state. Joining today's
squad value to a 2013 row is the same error as joining a contract expiry date,
which this project already bans. `clubs.csv` is the one file of the ten that is
deliberately not downloaded.

## Baselines

Gradient boosting, test-set metrics in EUR, from `scripts/train_baseline.py`.
The **temporal** row is the headline: it is the only split where the model has
never seen the season it is asked about, which is the only situation it will
meet in use.

| Split | performance-only | + prior value |
|---|---|---|
| Random (flattering) | R² 0.529 / MAE €2.24M | R² 0.784 / MAE €1.49M |
| Group by player | R² 0.604 / MAE €2.13M | R² 0.812 / MAE €1.69M |
| **Temporal** | **R² 0.762 / MAE €2.39M** | **R² 0.892 / MAE €1.75M** |

The gap between the random and temporal rows is the point. Reporting the random
number would roughly halve the stated error and answer a question nobody
deploying this will ever ask.

Everything is seeded from one constant; two runs agree exactly, and a test
asserts it. The leakage checks re-run after every split, because splitting is
what creates the chance of a row or a player straddling the boundary.

## The model zoo

Eleven families (Linear, Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees,
HistGradientBoosting, XGBoost, LightGBM, CatBoost, and a ridge-blended stack of
the three boosters), each searched on expanding-window folds inside the training
seasons, then scored once on the test seasons. LightGBM ships for both variants.

| Variant | Winner | Test MAE | Test R² | vs. baseline |
|---|---|---|---|---|
| performance-only | LightGBM | €2.21M ± 0.04M | 0.813 | 0.762 |
| with prior value | LightGBM | €1.66M ± 0.04M | 0.914 | 0.892 |

![The model page: held-out metrics, provenance, and every family ranked by validation MAE](docs/img/model.png)

*`/model` shows the winner's held-out metrics, its full provenance, and the
families it beat — with the spread between them stated rather than hidden.*

**"Wins" is doing less work than it looks.** The top four families on the
performance-only variant were separated by EUR 44,000 of validation MAE, and the
standard error of that MAE is EUR 60,000. They are the same model as far as this
data can tell.

So selection is not `min()`. Two rules decide what ships:

**It must be explainable.** Every prediction response documents an
`explanation`, the dashboard draws a contribution chart from it, and both model
cards are built from named importances. The stacked blend had the lowest
validation MAE on both variants and exposes no importances at all — its members
have them, the blend does not. HistGradientBoosting is subtler and shipped once
before this rule existed: SHAP works on it, so every prediction looked
explained, while `feature_importances_` has never existed on it and the model
card came out blank. Both are excluded, both still run, and the leaderboard
shows what excluding them costs: 1.92% and 0.33% of validation MAE.

**Then the one-standard-error rule** (Breiman, Friedman, Olshen & Stone, 1984):
among families within one standard error of the best, take the cheapest.
XGBoost beat LightGBM by EUR 7,917 — a seventh of one standard error, paired
t = 0.37 — and shipping it would have added 372 MB to the serving image, 291 MB
of that being CUDA libraries a CPU inference path never opens. The tiebreak is
deployment footprint rather than artifact size, because artifact size points the
wrong way: CatBoost serialises to a fifth of LightGBM's file and costs 328 MB of
package.

**The zoo still barely beats the baseline** — R² 0.762 → 0.813 and 0.892 →
0.914. Eleven families and a hyperparameter search bought about 0.05 and 0.02,
which is worth stating plainly rather than burying: the signal in this data is
in the features, not in the estimator. Phase 15 moved R² from 0.441 to 0.813 by
adding features, not by adding models.

Each run writes a versioned artifact to `models/` — the fitted preprocessing
and estimator as one object, plus metrics, named feature importance, the full
leaderboard and the seed — with a readable JSON sidecar. A test asserts a
reloaded artifact reproduces its recorded metrics exactly. The contract is
[`docs/EXPERIMENT_TRACKING.md`](docs/EXPERIMENT_TRACKING.md).

## Evaluation and explanations

`scripts/build_reports.py` reads the saved models — it trains nothing — and
writes self-contained HTML into `reports/`, plus a model card per variant and
[`docs/model_comparison.md`](docs/model_comparison.md), which shows *why*
LightGBM was selected rather than just that it was:

| | |
|---|---|
| `reports/baseline_report.html` | every family, accuracy beside training time, prediction time and model size |
| `reports/evaluation.html` | test metrics, predicted-vs-actual, residuals |
| `reports/feature_importance.html` | what the fitted model leans on |
| `reports/shap_summary.html` | global impact plus worked per-player waterfalls |
| `reports/error_analysis.html` | error by value band, age, position and season |

Two things the cost columns showed that accuracy alone would not: Random Forest
serialises to **458 MB** against LightGBM's 1.7 MB while scoring worse, and
CatBoost is within **0.75%** of the winner for 5.7× faster training and 4.8×
less disk — so if serving cost ever matters, that is the switch to make.

**Explanations are data, not pictures.** `src/explainability/` returns
dataclasses and floats and imports no plotting library, because Phase 9's
`POST /predict` has to return contributions as JSON and Phase 10 has to draw
them in a browser. The HTML is a rendering layer on top of the same functions.

SHAP values are additive in **log space**, since the models fit `log1p(EUR)`.
They are *not* additive in euros — the same contribution is worth a different
number of euros for a €500k player and a €90M one — so each one also carries
an exact multiplicative reading (`effect_multiplier`), which is the honest unit
for a log-target model and the one the API surfaces.

## API

```bash
make serve        # uvicorn api.main:app --reload
```

Interactive docs at `/docs`, the schema at `/api/v1/openapi.json`. The contract
is written down first, in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md), and
the service conforms to it — where the two disagree, the document is the defect
report.

All eleven, in full — a table that lists nine of eleven is worse than no table,
because a reader trusts it:

| Method | Path | What it answers |
|---|---|---|
| `GET` | `/health` | Alive, and can it predict? Unversioned, for orchestrators. |
| `POST` | `/api/v1/predict` | What is this player worth, and why? |
| `GET` | `/api/v1/players` | Which players match this name? |
| `GET` | `/api/v1/players/{player_id}` | Every season on file, plus a what-if seed. |
| `GET` | `/api/v1/players/{player_id}/similar` | Whose season most resembles this one? |
| `GET` | `/api/v1/players/{player_id}/history` | Predicted against actual, season by season. |
| `GET` | `/api/v1/features/distribution` | Where does a value sit in the population? |
| `GET` | `/api/v1/models` | Which variants are loaded? |
| `GET` | `/api/v1/models/{variant}` | Family, params, features, training data. |
| `GET` | `/api/v1/models/{variant}/metrics` | Held-out metrics, in EUR. |
| `GET` | `/api/v1/models/{variant}/feature-importance` | What the model leans on, and global SHAP. |

`tests/unit/test_api_contract_sync.py` diffs this table, `docs/API_CONTRACT.md`
and the live OpenAPI document against each other in every direction. The audit
that prompted it found two endpoints served for a week that the contract had
never heard of, then found the same two missing from this table after the
contract was fixed — a document nothing checks is checked once, on the day it
is written.

`POST /api/v1/predict` takes **either** a `player_id` **or** an explicit
`features` map — never both, never neither. Unknown feature names are rejected
rather than dropped, because silently ignoring a misspelt `minutes_playd`
returns a confident answer to a question nobody asked. Feature *values* are
checked too: a non-number, a non-finite number, a negative goal count or an
object where a scalar belongs comes back as a `422` naming the offending
feature, not as a `500`.

**On `confidence`.** A gradient booster has no calibrated uncertainty, and this
API will not invent one. The `confidence` field is an *empirical prediction
interval*: the model's own residual quantiles, measured on held-out seasons,
for the value band the prediction falls into, with the row count it was
measured over. `level: 0.8` means 80% of held-out predictions in that band
landed inside those bounds. The intervals are wide — roughly ×0.35 to ×5.0.
That is the honest finding, not a defect to tune away.

**Layering.** `src/services/prediction.py` imports no web framework — a test
asserts it, against parsed imports rather than a grep. That is what lets the
same prediction path serve HTTP, a batch job or a CLI, and why the 83 tests for
prediction logic need no running server.

## Dashboard

```bash
make serve                          # API on :8000
cd frontend && npm ci && npm run dev # dashboard on :3000
```

Next.js 16, React 19, Tailwind v4, Plotly. The flow is the product: search a
player, get a prediction with its interval, see what drove it, see comparable
seasons, then **change the inputs and watch the value move**.

| Page | What it answers |
|---|---|
| `/players` | Which player? Modellable ones first. |
| `/players/[id]` | What is he worth, why, next to whom — and what if? |
| `/compare` | How do two players differ, and on what? |
| `/analytics` | How is value distributed across the panel? |
| `/model` | How good is this model, and why was it chosen? |
| `/about` | Where the data came from and where not to trust it. |

![Two players compared: predictions, side-by-side contributions, and a percentile radar](docs/img/compare.png)

*`/compare` puts two players through the same model on the same basis —
contributions side by side, then a percentile radar across the whole panel.*

Every prediction is shown beside the model that produced it — family, training
date, dataset size, artifact version, and the **temporal** MAE and R². A number
with no attribution invites more trust than it has earned.

```bash
cd frontend && npm test        # 77 tests, ~2s
```

The suite covers the parts of the dashboard where being wrong is silent: the
error mapping in `lib/api.ts` (every failure becomes one `ApiError` carrying
the server's own `code`), the race guard in `useAsync` (navigate mid-request
and the stale response must not overwrite the fresh one), the money formatting,
and the state each component renders — including that a per-feature euro figure
never appears next to a SHAP contribution, because that number would be false.

It does not test what Plotly draws. jsdom cannot answer that, and the frontend
failure that actually breaks this app is the SSR trap, which `next build`
catches in CI.

Charts are not only pictures. Plotly emits an SVG of positioned shapes, so a
screen reader would otherwise get a label and no numbers — and the numbers are
the content. `Chart.tsx` marks the SVG `aria-hidden` and renders the same
series as a visually-hidden table, derived from the one `data` prop so the two
cannot describe different charts.

Three frontend traps, each handled in exactly one place:

- **Plotly and SSR.** A plain import fails `next build` with
  `ReferenceError: self is not defined` even inside `"use client"`. The dynamic
  import with `ssr: false` lives only in `src/components/Chart.tsx`, so every
  chart inherits the fix.
- **Bundle size.** The slim cartesian build is used, not full Plotly: 1.4 MB in
  its own lazy chunk rather than 4 MB. `@types/react-plotly.js` is deliberately
  not installed — v4 ships its own.
- **Tailwind v4 is CSS-first.** There is no `tailwind.config.js`; creating one
  would silently do nothing.

## Running it in containers

```bash
docker compose up --build       # API on :8000, dashboard on :3000
```

The API image installs `requirements-serve-lock.txt`, not the full stack: the
zoo trains nine families and serving loads one, and the difference is 291 MB of
CUDA libraries that xgboost brings for a GPU this inference path never touches,
plus 269 MB of catboost and its plotly. Dropping both took the image from
**2.62 GB to 1.32 GB** with an identical prediction on the same input. The
trade is stated in `requirements-serve.txt` and guarded by
`tests/integration/test_serving_dependencies.py`, which fails if a tuning run
ever ships an artifact the serving image cannot unpickle. `scripts/` is left
out of that image for the same reason — commands that would raise ImportError
are worse than commands that are absent.

The images deliberately **do not contain the data or the models**. The parquet
panel is hundreds of megabytes and refreshes weekly; the models are regenerated
by `scripts/train_models.py`. An image with either baked in is stale the day it
is built, so compose mounts `data/processed/` and `models/` read-only instead.
Run the three pipeline commands above at least once before `docker compose up`,
or the API starts *degraded*: `/health` returns 200 and reports
`ready: false`, which is the honest answer for a process that is alive but has
nothing to serve.

`NEXT_PUBLIC_API_BASE` is baked into the dashboard at **build** time, not run
time — Next inlines `NEXT_PUBLIC_*` into the client bundle. A deployment
pointing at a different API rebuilds the image. That is the real consequence of
shipping a URL to the browser, so it is stated rather than hidden behind a
runtime variable that would not work.

Postgres is not in the compose file. The storage layer sits behind a Protocol
(`src/storage/base.py`), so adding it later is a second implementation rather
than a rewrite — and running a database nothing reads from yet would be a
service to maintain for no measured benefit.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs four jobs on every
push and pull request:

- **Backend** — ruff, black, mypy, and the unit suite.
- **Dashboard** — tsc, eslint, and `next build`, the step that catches the
  Plotly SSR trap.
- **Authorship** — no `Co-Authored-By` trailers, no AI attribution lines.
- **Removed APIs** — greps for APIs that were deleted from dependencies this
  project pins, plus two architectural boundaries: `duckdb` may only be
  imported inside `src/storage/`, and Plotly only inside `Chart.tsx`.

CI runs on a clean checkout, where `data/` and `models/` are empty. Every test
that needs them **skips** rather than fails — verified on a fresh clone: 657
pass, the 49 integration tests skip, nothing fails. A suite that is only green on a machine with a trained
model is not a suite anyone can trust.

## Development

```bash
make help          # list every target
make test          # unit tests
make quality       # ruff + black --check + mypy
```

Python 3.13.

Dependencies are declared in two files on purpose. `requirements.txt` carries
reasoned ranges, each with a comment saying why the bound exists — `pandas<3`
because pandas 3 changes the default string dtype and makes Copy-on-Write
permanent, which is a silent behaviour change in a categorical-heavy pipeline.
`requirements-lock.txt` pins all 55 packages, transitives included, and is what
CI, the Docker image and `make setup` install, so a build today and a build in
March contain the same bytes. Regenerate it with:

```bash
uv pip compile requirements.txt --python-version 3.13 --output-file requirements-lock.txt
```

`tests/unit/test_dependencies.py` fails if the two disagree, if the lock stops
pinning exactly, if it steps over a declared bound, or if an HTML parser
reappears through a transitive dependency.

## Contributors

This project has a single author, Vansh Tomar. `scripts/hooks/commit-msg`
enforces it: commits carrying a `Co-Authored-By` trailer or an AI-generation
marker are rejected, as are commits authored by anyone else. The hook is
version-controlled and wired via `core.hooksPath`, so it survives a reclone.

## What this still cannot do

Stated because a model whose limits are not written down is a model whose
limits are discovered by whoever trusts it first.

**Coverage begins in 2012, and cannot be extended from this dataset.**
`games.csv` reaches back to 2006, which looks like five extra seasons. It is
not: before 2012-07-03 there are 2,470 events across 190 games and 943 players
— international tournaments, not league football — and `game_lineups.csv` does
not start until 2013. Minutes cannot be reconstructed from substitution events
in principle, because a player who plays the full match generates no event.
Career-length features are left-censored and capped at 10 for that reason.

**The labels are Transfermarkt's community estimates, not prices anyone paid.**
`transfers.csv` carries real fees and `--target transfer_fee` will train on
them, but it covers 8.5% of the table against the market value's 83%, and only
players who were actually sold — so it learns what a sold player costs, which
is not what a player is worth. The default target is the appraisal, and the
model reproduces that consensus including wherever it is biased.

**Error grows with value.** The target spans four orders of magnitude and the
headline MAE is in EUR, so a mid-table figure conceals much larger absolute
misses at the top of the market. Read the per-band breakdown in
`reports/error_analysis.html` before trusting a number for an expensive player.

**The prediction intervals are wide, and honestly so.** A gradient booster has
no calibrated uncertainty; the interval is measured from the model's own
residual quantiles on held-out seasons. Wide is the finding, not a defect to
tune away.

**The best model does not ship.** A stacked blend of the three boosters beat
every explainable family on both variants, by 1.92% and 0.33% of validation
MAE. It cannot produce a feature importance or a SHAP value, and every
prediction response documents an explanation, so it is excluded by
`EXPLAINABLE_REQUIRED` in [`src/pipelines/tune.py`](src/pipelines/tune.py). It
still runs and still appears in the leaderboard, so the cost of that decision
is visible rather than assumed.

**Seasons are August to July**, so leagues on a spring–autumn calendar are
split across that boundary and represented less faithfully.

## How it was built

Thirteen phases, each leaving the repository runnable, green and committed. The
plan and the decisions that changed on contact with the data are in
[`plans/IMPLEMENTATION_PLAN.md`](plans/IMPLEMENTATION_PLAN.md). Two are worth
reading on their own: the optional FBref enrichment was spiked and **declined on
measured evidence** ([`plans/02-fbref-spike.md`](plans/02-fbref-spike.md)), and
the release audit is recorded with its findings in
[`plans/03-final-verification.md`](plans/03-final-verification.md).

Backend test coverage, with the command that prints each number:

| | Coverage | Tests |
|---|---|---|
| `make test-cov` — no credentials, no data | **90%** | 657 pass, 49 deselected |
| `make test-cov-all` — needs the Kaggle download and a trained model | **97%** | 706 pass |

The gap is the pipeline orchestration in `src/pipelines/`, which is what the
integration tests exercise. Both targets fail below their floor, so neither
number can drift without CI going red. The frontend is covered separately by
`cd frontend && npm test` (see [Dashboard](#dashboard)).

## Licence

MIT — full text in [`LICENSE`](LICENSE).

The data carries its own terms and they are not MIT. Labels and appearances come
from `davidcariboo/player-scores`, which is CC0. Transfermarkt's own Terms of Use
§11.1 prohibit both automated access and ML training on the content, which is why
nothing here fetches it.
