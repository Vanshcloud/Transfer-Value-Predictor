# Transfer Value Predictor — Implementation Plan

Every phase cites `plans/00-discovery.md` (verified facts) rather than re-deriving them.
**Read that document before starting any phase.** Each phase leaves the repo runnable,
green, and committed.

**Authorship rule (applies to every commit in every phase):** Vansh is the sole author.
No `Co-Authored-By` trailer, no "Generated with" line, no AI attribution anywhere in a
commit message. Phase 1 installs a hook that makes this mechanical rather than
remembered.

---

## Decisions taken, and why they depart from the original brief

The brief was written before the data was examined. Four items changed on contact with
evidence; Vansh reviewed and adjusted them on 2026-08-28. Current standing decisions:

| Brief said | Plan does | Why (evidence in 00-discovery.md) |
|---|---|---|
| Scrape Transfermarkt, respecting ToS | **Never scrape it** | ToS §11.1 bans scraping *and* bans ML training on the content. Respecting the ToS means not doing it. CC0 Kaggle mirror used instead. |
| FBref as a core feature source | **Optional enrichment, Phase 12** | FBref is 403-Cloudflared; `soccerdata` emits no player ID so the join degrades to name matching; upstream issue #967 reports this exact call broken. `appearances.csv` gives per-match stats on the *same* `player_id` as the label. Retained as optional, not removed. |
| StatsBomb as a source | **Placeholder module, retained** | Aggregated stats are paywalled and coverage doesn't overlap the label panel, so it is not wired into the pipeline — but `src/ingestion/statsbomb.py` ships conforming to the ingestion Protocol so future expansion needs no refactor. |
| PostgreSQL + Redis | **DuckDB + Parquet now, Postgres-ready** | The data layer sits behind a storage Protocol from Phase 3, so Postgres is an added implementation, not a refactor. Redis waits for a measurement that justifies it. |

Adjustments requested by Vansh, incorporated below:

1. **Ingestion is `requests`-based wherever possible** — Playwright only if a source
   leaves no alternative, and never Selenium. (Phase 3)
2. **Storage abstraction from the start** so PostgreSQL slots in later without a rewrite.
   One `Protocol`, one DuckDB implementation — thin, not a framework. (Phase 3)
3. **Three-way temporal train/validation/test split** as the primary evaluation
   strategy. (Phase 6)
4. **An explicit leakage-detection stage** in preprocessing, not just tests. (Phase 4)
5. **Plotly, not Recharts** — this is an analytics project and Plotly's interactivity is
   the point. Cost is managed with the slim cartesian bundle (4028 KB → 1412 KB) behind
   `next/dynamic`, not by switching library. (Phase 10)

## Phase 1 — Repository foundation and the authorship guard

**Goal:** a real git repo *here*, quality gates that run, and a mechanical guarantee about
commit authorship.

**Implement**
1. `git init` in this directory. Confirm `git rev-parse --show-toplevel` returns this
   folder, **not** `$HOME` (see 00-discovery.md §4 — there is a stray empty
   repo in `$HOME`).
2. `.git/hooks/commit-msg`, executable, rejecting the message if it matches
   `Co-Authored-By|Generated with|Claude` (case-insensitive). Mirror it into
   `scripts/hooks/commit-msg` and have `make setup` install it, so it survives a reclone.
3. `pyproject.toml`: `requires-python = ">=3.13,<3.14"`, setuptools backend,
   `dependencies = {file = ["requirements.txt"]}`, packages `["src*"]`, author Vansh
   Tomar <vanshwar@gmail.com>. Ruff + Black + MyPy config.
4. `requirements.txt` with **`pandas>=2.2,<3`** (00-discovery.md §3 — unpinned resolves to
   pandas 3.0.5 and silently changes string dtypes) and `requirements-dev.txt`,
   `requirements-lint.txt` **pinned** (the sibling repo was bitten by unpinned lint tooling).
5. `.gitignore` — mirror the sibling repo's: ignore `data/raw/*`, `data/processed/*`,
   `models/*.joblib`, `.env`, keep `.gitkeep` and `!.env.example`.
6. `Makefile` with `help setup install install-dev test test-cov lint format format-check
   typecheck quality clean` — same target names as the sibling repo.
7. Directory skeleton from the brief, each with `.gitkeep`.
8. `README.md` stub stating what the project is and that **`brew install libomp` is
   required on macOS** (00-discovery.md §3 — lightgbm *and* xgboost both need it).
9. One trivial test so `make test` is green from commit 1.

**Verify**
- `git rev-parse --show-toplevel` → this directory.
- Attempting `git commit -m "test\n\nCo-Authored-By: Claude <x@y>"` is **rejected**.
- `make quality` and `make test` both pass.
- `git log --format='%(trailers:key=Co-Authored-By)' | grep -c .` → `0`.

**Do NOT**
- Run `git add .` before confirming the repo root. It would stage `$HOME`.
- Leave the hook only in `.git/hooks/` — it is not version-controlled there.

---

## Phase 2 — Config, logging, and typed utilities

**Implement**
- `configs/config.yaml` + `.env.example`. Nothing hardcoded, no secrets in code.
- `src/utils/config.py` — load YAML, overlay env via `python-dotenv`, return a typed
  Pydantic settings object. **Pydantic v2 idioms only** (00-discovery.md §3:
  `model_config = ConfigDict(...)`, `@field_validator`, `.model_dump()`).
- `src/utils/logging.py` — structured logging, configured once.
- `src/utils/paths.py` — every path derived from a project root constant, no `../..`.

**Verify** `make quality` clean; a test asserts env overrides YAML; no
`DeprecationWarning` from pydantic under `pytest -W error::DeprecationWarning`.

**Do NOT** use `@validator`, `.dict()`, or `class Config` — all deprecated and warned.

---

## Phase 3 — Ingestion

**Implement**
- `src/ingestion/base.py` — a `Protocol` defining the standardized frame contract every
  provider returns. One `Protocol`, not an ABC hierarchy.
- **All HTTP via `requests`** with a shared `requests.Session`: retry, exponential
  backoff, timeout, configurable rate limit, custom User-Agent. No Playwright in this
  phase; never Selenium.
- `src/ingestion/statsbomb.py` — **placeholder** conforming to the Protocol, raising
  `NotImplementedError` with a message pointing at 00-discovery.md §1.4. It exists so
  future expansion is an implementation, not a refactor. Keep it to a stub; do not build
  an event-aggregation pipeline nothing consumes yet.
- `src/storage/` — a `Protocol` (`read_table` / `write_table` / `query`) with a single
  DuckDB+Parquet implementation. Every downstream phase talks to the Protocol, never to
  DuckDB directly, so PostgreSQL is a second implementation later. One Protocol, one
  implementation, no factory.
- `src/ingestion/kaggle_loader.py` — download `davidcariboo/player-scores`
  (00-discovery.md §1.2), cache raw CSVs under `data/raw/`, convert to parquet in
  `data/processed/`. Skip download when the cache is fresh.
- Retry with exponential backoff, timeout, custom UA, structured logging.
- `scripts/fetch_data.py` CLI entry point.

**Verify**
- `python scripts/fetch_data.py` produces `players`, `player_valuations`, `appearances`
  parquet files. Row counts match 00-discovery.md §1.2 within a weekly-refresh delta
  (50,149 / 656,301 / 1,894,350).
- Re-running hits cache and makes no network call (assert via a mocked session).
- Tests use a committed fixture under `data/sample/`, never the network.

**Do NOT** write a Transfermarkt scraper (00-discovery.md §1.1). Do not import
`soccerdata` in this phase.

---

## Phase 4 — Validation

**Implement** `src/validation/` — reusable checks for missing values, duplicate
`(player_id, date)`, non-positive values, unparseable dates, unknown positions,
out-of-range ages/heights. Returns a structured report; raises only on contract violations.

**Leakage detection stage** — `src/validation/leakage.py`, run as an explicit pipeline
step, not only as a test. It must fail loudly on:
- any feature column whose source timestamp is later than the row's label date;
- current-state columns attached to historical rows (`contract_expiration_date` is the
  known offender — 39.5% null and *current*, so it leaks the future into every past season);
- target-derived columns reaching the feature matrix unlagged;
- train/test index overlap after splitting.
Emits a structured report; raises on violation. Wire it into the training pipeline so a
leak stops a run rather than quietly inflating a metric.

**Verify** Tests must cover the two traps found in the spike (00-discovery.md §2.2):
1. `position == "Missing"` is a **literal string**, not `NaN` — a check using only
   `isna()` passes over 37 real rows. Assert it is caught.
2. `player_valuations` primary key is `(player_id, date)`, not `player_id`.

---

## Phase 5 — Feature engineering and the training table

This is the phase the whole project rests on. The spike in `scratchpad/FINDINGS.md`
already proved it works end to end — port it, don't reinvent it.

**Implement** `src/feature_engineering/`:
- Season assignment: Aug–Jul (`month >= 8 ? year : year - 1`).
- Aggregate `appearances` → player-season: apps, goals, assists, minutes, cards.
- **Label via `pd.merge_asof(..., direction="forward", tolerance=pd.Timedelta("120D"))`**
  on the first valuation *after* season end. Never an equality merge — valuations are an
  irregular on-change event series (00-discovery.md §1.2).
- Join `players` for age, position, foot, height, nationality.
- Derived: per-90 rates, minutes-per-appearance, age curve terms.
- Prior-season market value as an **explicitly lagged** feature, isolated behind a flag so
  the two model variants (§2.3) can be built from one code path.

**Verify**
- Output ≈ **37,025 rows / 16,995 players / seasons 2011–2024** (±weekly refresh drift).
- With prior value: ≈ **20,030 rows**.
- Null rates match 00-discovery.md §2: age 0.0%, position 0.0%, minutes 0.0%, height 1.2%.
- A test asserts no row's feature timestamp is later than its label date.

**Do NOT**
- Use `contract_expiration_date` from `players.csv` on historical rows. It is **current**
  state (39.5% null) and leaks the future into every past season. Drop it, or re-derive
  per-season from `transfers.csv`.
- Compute per-90 rates without clipping the minutes denominator.

---

## Phase 6 — Baseline model and split discipline

Establish the honest number *before* building the zoo, so the zoo has something to beat.

**Implement**
- `src/models/splits.py` — a **three-way temporal train/validation/test** splitter as the
  primary strategy (train ≤2021, validation 2022, test 2023+), plus group-by-player and
  random splitters behind the same interface for diagnostic comparison only. Validation
  tunes; test is touched once. **Temporal is the reported headline.**
- Target transform `log1p` / inverse `expm1` (00-discovery.md §2: skew 8.70 → 0.43).
- `src/models/baseline.py` — Ridge and a single gradient-boosting baseline.
- Metrics in **EUR**, not log space, via `root_mean_squared_error` (00-discovery.md §3 —
  `mean_squared_error(squared=False)` now raises `TypeError`).

**Verify** reproduce the spike table within noise:

| Split | Baseline | + prior value |
|---|---|---|
| Random | R² ~0.465 | R² ~0.826 |
| Group | R² ~0.455 | R² ~0.830 |
| **Temporal** | **R² ~0.412 / MAE ~€5.14M** | **R² ~0.809 / MAE ~€3.96M** |

Seed everything; two runs must agree exactly. (The sibling project shipped 14 days of
unreproducible metrics before catching a missing seed — don't repeat it.)

**Do NOT** report random-split metrics as headline. Temporal MAE is ~60% worse and is the
number that reflects deployment.

---

## Phase 7 — Model zoo, tuning, selection

**Implement** all nine models from the brief (Linear, Ridge, Lasso, ElasticNet, RF,
GradientBoosting, XGBoost, LightGBM, CatBoost) behind one registry interface; CV, tuning,
automatic best-model selection by temporal-split MAE. Persist model + metrics + feature
importance + config + the fitted preprocessing pipeline as one versioned artifact.

**Verify** every model trains; the selection is reproducible; the saved artifact reloads
and reproduces its recorded metrics exactly.

**Do NOT** use `OneHotEncoder(sparse=...)` — removed, raises `TypeError`. Use
`sparse_output=`. Use `.set_output(transform="pandas")` so feature names survive into SHAP.

---

## Phase 8 — Evaluation and explainability

**Implement** MAE/RMSE/R²/MAPE; residual, predicted-vs-actual, error-distribution and
feature-importance plots (matplotlib for reports); SHAP integration returning per-prediction
positive/negative contributions.

**Verify** `shap.TreeExplainer(model)(X)` returns an `Explanation`; a waterfall renders.

**Do NOT** pass raw ndarrays to `shap.plots.waterfall` — it raises `TypeError` and
requires an `Explanation` (00-discovery.md §3). Use `explainer(X)`, not
`.shap_values(X)`.

---

## Phase 9 — FastAPI backend

**Implement** `GET /health`, `POST /predict`, `GET /player/{id}`, `GET /model/info`,
`GET /metrics`, `GET /feature-importance`. Pydantic v2 request/response models. Model
loaded once in a **lifespan** handler and injected with `Depends`.

**Verify** every endpoint under `TestClient`; a malformed payload returns 422 with a
useful body; `/predict` returns SHAP contributions.

**Do NOT** use `@app.on_event("startup")` — deprecated (00-discovery.md §3). Do not load
the model at import time.

---

## Phase 10 — Frontend

**Implement** scaffold with the verified command (00-discovery.md §3), then Home, Player
Search, Prediction, Comparison, League Analytics, Model Metrics, About. Dark mode,
loading and error states, responsive.

**Verify** `next build` passes — this is where a naive Plotly import fails with
`ReferenceError: self is not defined`.

**Charting is Plotly** (analytics project — its hover/zoom/selection is the point).
Wrap it once in `src/components/Chart.tsx` using the **slim cartesian bundle** via
`react-plotly.js/factory`, loaded through `next/dynamic` with `ssr: false` inside a client
component. Every chart imports that wrapper, so the SSR guard and bundle choice exist in
exactly one file.

**Do NOT** create `tailwind.config.js` — Tailwind v4 is CSS-first via `@theme`. Do not
install `@types/react-plotly.js` — v4 ships its own types. Do not import `plotly.js`
directly in a page: a plain import fails `next build` with `ReferenceError: self is not
defined` even inside `"use client"`, and pulls 4028 KB.

---

## Phase 11 — Docker, CI, docs

**Implement** Dockerfile(s), docker-compose (API + frontend; Postgres only if adopted),
GitHub Actions for lint/test/build with **pinned** tooling, full README.

**Verify** CI green on a clean checkout with `data/raw/` and `models/` absent — tests that
need a trained model must **skip**, not fail. Docker daemon is currently stopped; start it
before this phase.

---

## Phase 12 — FBref enrichment (optional, gated)

**Spike first, decide second.** Install `soccerdata`, attempt one
`read_player_season_stats(stat_type="standard")` call with `headless=True`. Upstream issue
#967 says this is currently broken; if it fails, stop and use the MIT-licensed
`hubertsidorowicz/football-players-stats-2024-2025` Kaggle mirror instead.

Only if it works: name-based join via the latin-1 crosswalk, disambiguated by birth year
and nation, with a **measured coverage report**. Accept the enrichment only if temporal
MAE improves on the Phase 6 baseline.

**Do NOT** let this phase block anything. **Do NOT** assume >80% join yield — that figure
is ID-to-ID; the name-matching hop degrades it by an unmeasured amount.

---

## Phase 13 — Final verification

- `make quality`, full test suite, Docker build, CI all green.
- `git log --format='%(trailers:key=Co-Authored-By)' | grep -c .` → **0**.
- `git log --format='%an' | sort -u` → **Vanshcloud only**.
- Grep the tree for the removed APIs: `mean_squared_error(.*squared=`,
  `OneHotEncoder(.*sparse=`, `@app.on_event`, `@validator`, `\.dict()`,
  `tailwind.config.js`, `from ["']plotly.js["']` → all zero hits.
- Storage: no phase outside `src/storage/` imports `duckdb` directly.
- Every number in the README traceable to a command that regenerates it.
- Confirm no code path fetches transfermarkt.com.
