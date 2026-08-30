# Phase 0 — Discovery Findings (verified 2026-08-28)

Everything here was **executed or fetched**, not recalled. Each phase of the
implementation plan cites this document instead of re-deriving facts.
Anything not verified is labelled UNVERIFIED.

---

## 1. Data sources — the decisive findings

### 1.1 Transfermarkt direct scraping is OFF the table

`transfermarkt.com/robots.txt` (fetched, HTTP 200) is permissive — `User-agent: *` /
`Allow: /`, no `Disallow`, no `Crawl-delay`. **The Terms of Use are not.**
`transfermarkt.com/intern/anb` §11.1, verbatim:

> The User is not permitted to access or copy the Digital Content using bots,
> spiders, screen scraping or other automated processes. The user is also
> prohibited from using the digital content for the training or development of
> artificial intelligence (AI), including language models, machine learning,
> neural networks or other AI systems. Uses for text and data mining
> (Section 44b UrhG) are expressly reserved.

That clause bans the *method* and, separately, the *exact purpose of this project*.
The final sentence is the §44b UrhG opt-out, which disables the TDM exception that
would otherwise be the fallback defence. §3.2 asserts database rights; §10 puts it
under German law / Hamburg courts.

**Decision: never scrape transfermarkt.com.** This honours the brief's own
instruction to respect terms of service. No `src/ingestion/transfermarkt.py`
scraper will be written.

### 1.2 Kaggle `davidcariboo/player-scores` — the foundation (CC0)

Verified via authenticated Kaggle API **and by downloading and counting the files**.

| Fact | Value |
|---|---|
| Slug | `davidcariboo/player-scores` |
| License | **CC0: Public Domain** |
| Updated | 2026-08-05, version 677, weekly cadence |
| Size | ~759 MB, 12 CSVs |

Real headers, from the downloaded files:

```
player_valuations.csv: player_id,date,market_value_in_eur,current_club_name,
                       current_club_id,player_club_domestic_competition_id
players.csv:           player_id,first_name,last_name,name,last_season,current_club_id,
                       player_code,country_of_birth,city_of_birth,country_of_citizenship,
                       date_of_birth,sub_position,position,foot,height_in_cm,
                       contract_expiration_date,agent_name,image_url,international_caps,
                       international_goals,current_national_team_id,url,
                       current_club_domestic_competition_id,current_club_name,
                       market_value_in_eur,highest_market_value_in_eur
appearances.csv:       appearance_id,game_id,player_id,player_club_id,player_current_club_id,
                       date,player_name,competition_id,yellow_cards,red_cards,goals,
                       assists,minutes_played
```

Measured: `player_valuations` 656,301 rows / 41,528 players / 2000-01-20→2026-06-12.
`appearances` 1,894,350 rows / 29,531 players. `players` 50,149 rows.

**`appearances.csv` carries the same `player_id` as the label.** Features and labels
join on an integer key with no entity resolution at all.

`player_valuations` rows appear *on change*, not on a schedule — it is an irregular
event series and must be as-of joined, never merged on equality.

### 1.3 FBref / `soccerdata` — optional enrichment, NOT a foundation

Real signature from the 1.9.1 sdist:

```python
soccerdata.FBref(leagues=None, seasons=None, proxy=None, no_cache=False,
                 no_store=False, data_dir=..., path_to_browser=None, headless=False)
FBref.read_player_season_stats(stat_type: str = "standard") -> pd.DataFrame
# stat_type ∈ {"standard","keeper","shooting","playing_time","misc"}  — only 5
# index: MultiIndex[league, season, team, player]
```

Blockers, all verified:
- `fbref.com` and `sports-reference.com` return **HTTP 403 Cloudflare** to non-browser
  clients — their robots.txt is itself unreadable.
- `soccerdata` **has no FBref player ID** (`player_id` appears in its understat/sofifa/
  whoscored modules, never `fbref.py`). Upstream issues #728, #806, #827 request it; all open.
- Its changelog advertises *"bypass Cloudflare protection"* and *"CAPTCHA solver"* —
  it circumvents access controls; that is a choice to make knowingly.
- `rate_limit = 7` seconds → ~8.5 req/min. Requires a real Chrome.
- Open issue **#967 (2026-08-05, unanswered)** reports `read_player_season_stats` failing
  with a CAPTCHA error — the exact call this project would make.
- Docstring says `headless=True`; the actual default is `False`.

A crosswalk exists — `JaseZiv/worldfootballR_data` →
`raw-data/fbref-tm-player-mapping/output/fbref_to_tm_mapping.csv`, 15,440 rows,
**latin-1 encoded**, last updated 2025-06-21 (~14 months stale). Measured overlap with
our data: 12,253 TM ids (80.6%) have valuations. But it keys on *FBref player ID*, which
soccerdata does not emit — so the chain still degrades to a **name match**.

**Decision: FBref is a Phase 9 spike, gated on a working baseline.** Legal-risk-free
alternative if we want FBref-style metrics: `hubertsidorowicz/football-players-stats-2024-2025`
on Kaggle (MIT, pre-scraped).

### 1.4 StatsBomb — dropped, with reasons

`statsbombpy` 1.22.0. `player_season_stats()` **raises on open data** ("please provide
credentials") — the one model-ready function is paywalled. Open data is 80
competition-seasons skewed historical: Premier League has only **2003/04 and 2015/16**;
Champions League includes 1970/71. It does not overlap our label panel, it would need a
bespoke event→season aggregation, and it adds a third ID namespace.

**Decision: out of scope.** Excellent tactical data; wrong data for market value.

---

## 2. Feasibility spike — measured on the real data

A prototype table was built end to end (`scratchpad/FINDINGS.md`):

- **37,025 rows / 16,995 players / seasons 2011–2024**, 20,030 rows once a prior season exists.
- Target EUR: median 500k, p95 10M, max 200M, min 10k, **no nulls, no zeros**.
- **Skew: raw 8.70 → log1p 0.43.** Train on `log1p(value)`; report EUR.
- Nulls: age 0.0%, position 0.0%, minutes 0.0%, height 1.2%,
  **`contract_expiration_date` 39.5%**.

### 2.1 Split strategy — measured, and it corrected a prior assumption

| Split | Baseline feats | + prior-season value |
|---|---|---|
| Random (leaky) | R² 0.465 / MAE €3,277,047 | R² 0.826 / MAE €2,393,800 |
| Group by player | R² 0.455 / MAE €3,243,155 | R² 0.830 / MAE €2,370,351 |
| **Temporal (≤2022/2023+)** | **R² 0.412 / MAE €5,140,804** | **R² 0.809 / MAE €3,958,350** |

The expected random-vs-group leak is **negligible** — the as-of join already yields one
row per player-season. The real gap is **temporal: ~60% worse EUR MAE**. Publish the
temporal number; anything else flatters the model.

### 2.2 Two anti-patterns this spike exposed

1. **`contract_expiration_date` in `players.csv` is CURRENT state**, not per-season.
   Attaching it to a 2015 row leaks the future. Drop it from historical rows.
2. **`position` uses the literal string `"Missing"`**, not `NaN` (37 rows). Null checks
   that only test `isna()` will pass over it.

### 2.3 A product decision the numbers force

Prior market value moves R² from 0.45 → 0.83 and dominates every other feature. Ship
**both** variants and label them honestly:
- **performance-only** — the useful model (scouting, undervalued-player detection)
- **with-prior-value** — the accurate model (tracking, forecasting)

Publishing only the second is technically true and practically useless.

---

## 3. Allowed APIs — verified signatures

**Runtime: Python 3.13.** All of scikit-learn 1.9.0, xgboost 3.4.1, lightgbm 4.7.0,
catboost 1.2.10, shap 0.52.0, pyarrow 25.0.1, fastapi 0.141.1, uvicorn 0.52.4,
pydantic 2.13.4 install from prebuilt wheels on 3.12/3.13/3.14 with zero source builds.
3.13 is chosen for wheel-coverage headroom.

**Pin `pandas>=2.2,<3`.** Unpinned it resolves to **pandas 3.0.5**, which changes the
default string dtype to `str` and makes Copy-on-Write permanent — silent dtype drift in a
categorical-heavy pipeline.

**macOS prerequisite: `brew install libomp`.** Both `lib_lightgbm.dylib` and
`libxgboost.dylib` link `@rpath/libomp.dylib` and bundle nothing; the failure was
reproduced as `OSError: dlopen ... Library not loaded: @rpath/libomp.dylib`. sklearn
bundles its own copy, which is why "sklearn works but lightgbm doesn't" is a common report.
Non-issue on Linux/Docker (manylinux wheels bundle libgomp) — belongs in the README.

### sklearn 1.9.0 — two removals that WILL break code

```python
# REMOVED, raises TypeError — not deprecated:
mean_squared_error(y, p, squared=False)
OneHotEncoder(sparse=...)

# CORRECT:
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)
Pipeline(steps, *, transform_input=None, memory=None, verbose=False)
ColumnTransformer(transformers, *, remainder="drop", sparse_threshold=0.3, ...)
```
Use `.set_output(transform="pandas")` so feature names survive into SHAP.

### pandas 2.3.3 + numpy 2.5.2 — `pd.Timedelta` idiom

Found in Phase 4 by the `-W error::DeprecationWarning` gate. Both of these warn:

```python
pd.Timedelta("30D")        # DeprecationWarning: 'generic' unit for NumPy timedelta
pd.Timedelta(days=30)      # same
```

The clean forms are `pd.Timedelta(30, "D")` or `pd.offsets.Day(30)`. This
matters for Phase 5, whose `merge_asof` label join uses a 120-day tolerance.

### shap 0.52.0

```python
explainer = shap.TreeExplainer(model)
sv = explainer(X)              # __call__ -> Explanation ; .shap_values() -> bare ndarray (legacy)
shap.plots.waterfall(sv[0])    # REQUIRES an Explanation; raises TypeError on ndarray
shap.plots.beeswarm(sv)
```
`shap.force_plot` / `summary_plot` / `waterfall_plot` still exist and emit **no**
deprecation warning — they are legacy aliases, not deprecated. Use `shap.plots.*` anyway.

### pydantic 2.13.4 — these emit real deprecation warnings

`@validator`→`@field_validator` · `@root_validator`→`@model_validator` · `.dict()`→`.model_dump()`
· `.json()`→`.model_dump_json()` · `class Config`→`model_config = ConfigDict(...)`
· `schema_extra`→`json_schema_extra` · `.parse_obj()`→`.model_validate()`

### fastapi 0.141.1 — `@app.on_event` is deprecated

```python
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["value_model"] = joblib.load(...)
    yield
    ml_models.clear()
app = FastAPI(lifespan=lifespan)
```
Inject with `Annotated[Model, Depends(get_model)]`. Note: `starlette.testclient` with
`httpx` now warns to install `httpx2` — a test-dependency concern.

### Frontend — Next.js 16.3.3 / React 19.2.8 / Tailwind 4.3.3

```bash
npx create-next-app@latest frontend --ts --tailwind --app --eslint --src-dir \
    --import-alias "@/*" --yes
```
**Tailwind v4 is CSS-first — there is no `tailwind.config.js`.** Config lives in
`@theme` blocks in CSS; PostCSS plugin is `@tailwindcss/postcss`.

**Charting:** `react-plotly.js` 4.1.0 is maintained again (revived 2026-06 after a
4-year gap) and ships its own types — do NOT install `@types/react-plotly.js`. But a
plain import **fails `next build`** with `ReferenceError: self is not defined` even inside
`"use client"`, because client components are prerendered. Measured bundle: **full
plotly.js = 4028 KB**; `plotly.js-cartesian-dist-min` via factory = **1412 KB**.

**Recommendation: Recharts 3.10.1 for standard dashboard charts** (SSR-safe, no dynamic
import, far lighter, composes with Tailwind); reach for slim Plotly only where a specific
chart needs its interactivity. If Plotly is used:
```tsx
"use client";
import dynamic from "next/dynamic";
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
```

---

## 4. Environment facts

Toolchain versions the plan depends on. Host-specific inventory that was here
during development — local paths, an account name, free disk — has been removed:
it explained nothing a reader of this repository needs.

Removed from the **history** as well, not only from this file. Deleting it in a
later commit leaves it in every earlier one, and the deleting diff renders it
again in its own `-` lines; the release audit found exactly that. The history
was rewritten with `git filter-repo` while the repository was still private with
no forks, and `.github/workflows/ci.yml` now greps every commit so it cannot
come back.

- Python 3.13, node 24 / npm 11. Both are what CI and the Dockerfiles pin.
- Kaggle credentials are required for ingestion and are supplied by the operator,
  never committed. See `.env.example`.
- A stray `git init` in `$HOME` was found before Phase 1 — hence Phase 1's first
  step verifying `git rev-parse --show-toplevel` resolves to the project
  directory. `git add .` from the wrong root would stage a home directory.
- `psql` is not required: the storage layer is DuckDB behind a Protocol.

## 5. UNVERIFIED — do not treat as fact

- FBref's "10 req/min, 1-hour block" figure came from search results; the policy page
  itself returns 403. The *existence* of aggressive limiting is certain.
- Whether `soccerdata`'s FBref scraper works **today**. Not installed. Issue #967 says no.
  Must be spiked before anything depends on it.
- Whether `transfers.csv` carries a fee column (repo schema shows none; file not downloaded).
- Linux/Docker wheel coverage (expected fine, untested).
- Recharts vs Plotly head-to-head bundle size (Plotly figures measured; Recharts inferred).
- The Next.js SSR fix is proven to **build**, not proven to **paint** in a browser.
