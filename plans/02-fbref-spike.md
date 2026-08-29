# FBref enrichment spike — measured, not assumed (2026-08-29)

Phase 12 says *spike first, decide second*. This is the spike. **Decision: no-go.**
Nothing was merged into the pipeline; no dependency was added to the project.

Reproduce any line below without touching the project environment:

```sh
uv run --python 3.13 --with soccerdata python -c "
import soccerdata as sd
fb = sd.FBref(leagues='ENG-Premier League', seasons='2023-24', headless=True)
print(fb.read_player_season_stats(stat_type='standard').shape)"
```

## 1. Upstream issue #967 is stale — the call works

`soccerdata` 1.9.1, py3.13. `read_player_season_stats(stat_type="standard")` for
ENG-Premier League returns data, it does not raise:

| season  | rows |
|---------|------|
| 2015-16 | 561  |
| 2017-18 | 529  |
| 2020-21 | 532  |
| 2023-24 | 580  |

The plan's stated blocker no longer reproduces. Three other reasons replace it.

## 2. The only transport that works is Selenium — which is banned

`fbref.com` returns **403 to plain `requests`**, with the project User-Agent and with a
current desktop Chrome one:

```
UA='transfer-value-predictor/0.1'  -> 403 Forbidden
UA='Mozilla/5.0 ... Chrome/140.0'  -> 403 Forbidden
```

soccerdata gets through by launching chromedriver. Standing decision #1
(IMPLEMENTATION_PLAN.md) is *"Playwright only if a source leaves no alternative, and
never Selenium"*, and `src/utils/http.py` states the same in its own docstring. Adopting
soccerdata means adopting Selenium plus ~70 transitive packages into a requests-only
ingestion layer.

Playwright would be the letter-of-the-rule escape hatch, since FBref genuinely leaves no
alternative. Section 3 is why that hatch is not worth opening.

## 3. The payload is what we already have, on a worse key

`stat_type="standard"` columns:

```
nation, pos, age, born,
Playing Time: MP, Starts, Min, 90s
Performance:  Gls, Ast, G+A, G-PK, PK, PKatt, CrdY, CrdR
Per 90 Minutes: Gls, Ast, G+A, G-PK, G+A-PK
```

Every one of those is already in `training_table.parquet`, derived from Kaggle
`appearances.csv` and joined on the **real integer `player_id`** — not on a name.

The reason to want FBref was the advanced signal: xG, xA, npxG, progressive passes,
carries, take-ons. Measured across four seasons, it is not reachable on this path. For
every season tried, soccerdata offers only:

```
stat_type should be in ['standard', 'keeper', 'shooting', 'playing_time', 'misc']
```

`passing` and `possession` are rejected outright — in 2023-24 as well as 2015-16. And
`shooting`, the one type that could carry xG, does not:

```
2020-21 shooting cols: nation, pos, age, born, 90s,
                       Standard: Gls, Sh, SoT, SoT%, Sh/90, SoT/90, G/Sh, G/SoT, PK, PKatt
```

No xG column in 2015-16, 2017-18, 2020-21 or 2023-24. fbref.com the website has xG from
2017-18; the library path we are permitted to use does not surface it.

Net new signal from the whole exercise: **shots and shots-on-target**. Delivered by a
name-only join (soccerdata emits no player ID), disambiguated by birth year and nation,
at an unmeasured yield.

## 4. The documented fallback is structurally dead

Plan's fallback is the MIT mirror `hubertsidorowicz/football-players-stats-2024-2025`.
Confirmed via the Kaggle API — single season, 2024-25.

Our split (`configs/config.yaml`) is train ≤2021, validation 2022, test ≥2023. A
2024-25-only feature lands **entirely inside the test set**. The model never sees it in
training, so it cannot learn a coefficient for it; it can only add noise to the one split
we are allowed to touch once. It cannot improve temporal MAE — not "probably won't", but
cannot, by construction.

## Decision

Phase 12's acceptance gate is *"accept the enrichment only if temporal MAE improves on
the Phase 6 baseline."* The gate cannot be met:

- the working transport is banned, and the escape hatch buys shots-on-target;
- the columns duplicate an existing feature set that joins on an integer key;
- the fallback source cannot reach the training split at all.

Phase 12 closes as **decided, not deferred**. `src/ingestion/statsbomb.py` already stands
as the pattern for a source that conforms to the ingestion Protocol without being wired
in; FBref does not need a second such placeholder.

Reopen this only if soccerdata (or a successor) exposes `passing`/`possession` with xG
for pre-2022 seasons **and** a player ID. Both conditions, not either.
