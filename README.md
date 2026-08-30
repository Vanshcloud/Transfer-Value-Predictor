# Transfer Value Predictor

Predict the market value (EUR) of professional footballers from performance,
biographical and contextual data — and explain every prediction.

Search a player, get a valuation with a confidence interval, see the SHAP
contributions that produced it, compare him to the seasons nearest him in the
model's own feature space — then change the inputs and watch the number move.

Metrics are reported on **held-out seasons the model has never seen**: test MAE
**€4.44M** (R² 0.441) for the scouting model, **€3.71M** (R² 0.775) with a prior
valuation. A random split would read about 35% better and answer a question
nobody deploying this will ever ask.

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
so `make test` gives you 475 passing tests and 45 skips with no credentials at
all — the 45 are the integration tests that need the full panel, and they skip
rather than fail by design. Only the three pipeline commands below need the
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

The training table as of the 2026-08-05 dataset refresh:

| | |
|---|---|
| rows | 36,880 |
| players | 17,053 |
| seasons | 2011–2024 |
| rows with a prior-season value | 19,827 |
| leakage findings | 0 |

The label is the first valuation recorded **after** the season's evidence is
complete — an as-of join with a 120-day tolerance, never an equality merge,
because valuations are an irregular on-change event series. Two model variants
come out of the one table: *performance-only* (every row; the useful model, for
scouting) and *with prior value* (19,827 rows; the accurate model, for
tracking). Shipping only the second would be technically true and practically
useless.

## Baselines

Gradient boosting, test-set metrics in EUR, from `scripts/train_baseline.py`.
The **temporal** row is the headline: it is the only split where the model has
never seen the season it is asked about, which is the only situation it will
meet in use.

| Split | performance-only | + prior value |
|---|---|---|
| Random (flattering) | R² 0.499 / MAE €2.84M | R² 0.822 / MAE €2.35M |
| Group by player | R² 0.550 / MAE €2.60M | R² 0.786 / MAE €2.33M |
| **Temporal** | **R² 0.414 / MAE €4.52M** | **R² 0.766 / MAE €3.74M** |

The gap between the random and temporal rows is the point. Reporting the random
number would roughly halve the stated error and answer a question nobody
deploying this will ever ask.

Everything is seeded from one constant; two runs agree exactly, and a test
asserts it. The leakage checks re-run after every split, because splitting is
what creates the chance of a row or a player straddling the boundary.

## The model zoo

Nine families (Linear, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting,
XGBoost, LightGBM, CatBoost), each searched on expanding-window folds inside the
training seasons, selected by validation MAE in EUR, then scored once on the
test seasons. LightGBM wins both variants.

| Variant | Winner | Test MAE | Test R² | vs. Phase 6 baseline |
|---|---|---|---|---|
| performance-only | LightGBM | €4.44M | 0.441 | 0.414 |
| with prior value | LightGBM | €3.71M | 0.775 | 0.766 |

![The model page: held-out metrics, provenance, and every family ranked by validation MAE](docs/img/model.png)

*`/model` shows the winner's held-out metrics, its full provenance, and the nine
families it beat — with the spread between them stated rather than hidden.*

**The zoo barely beats the baseline** — R² moves 0.414 → 0.441 and 0.766 →
0.775. Nine families and a hyperparameter search bought about 0.03 and 0.01.
That is worth stating plainly rather than burying: the signal in this data is
in the features, not in the estimator, and an untuned gradient booster gets
most of the way there. Establishing the baseline first (Phase 6) is what makes
that measurable instead of assumed.

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
same prediction path serve HTTP, a batch job or a CLI, and why the 29 tests for
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
cd frontend && npm test        # 70 tests, ~1.5s
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
that needs them **skips** rather than fails — verified on a fresh clone: 475
pass, the 45 integration tests skip, nothing fails. A suite that is only green on a machine with a trained
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
| `make test-cov` — no credentials, no data | **89%** | 507 pass, 45 deselected |
| `make test-cov-all` — needs the Kaggle download and a trained model | **97%** | 552 pass |

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
