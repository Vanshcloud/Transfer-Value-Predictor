# Model card — performance_only

Generated 2026-08-29T09:07:53.371592+00:00 from `performance_only__lightgbm`.
This file is written from the artifact, so it cannot describe a model that
is no longer the one on disk.

## What it does

Estimate a player's market value from on-pitch performance and biography alone, with no prior valuation supplied. This is the scouting and undervaluation model: it is the one that can express an opinion that differs from the market, because it has never been told what the market thinks.

## Not what it is for

- Setting or negotiating an actual transfer fee. Market value and transfer fee are different quantities; this model is trained on the former.
- Players outside the covered competitions, or below the appearance volume the training data contains.
- Any individual decision about a person's employment or compensation without a human reviewing the explanation alongside the number.

## Model

| Family | `lightgbm` |
|---|---|
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.05, 'model__regressor__num_leaves': 63}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 19 |
| Seed | 42 |

## Data and split

- Rows: 36,880
- Split: temporal — train ≤2021, validation 2022, test ≥2023
- Source: Kaggle `davidcariboo/player-scores` (CC0). Transfermarkt is
  never scraped; its terms prohibit both the method and this purpose.

## Measured performance

Test seasons, in EUR. These are held-out seasons the model never saw.

| Metric | Validation | Test |
|---|---|---|
| MAE | €3,747,415 | €4,442,428 |
| RMSE | €9,104,983 | €10,518,792 |
| R² | 0.407 | 0.441 |
| MAPE | 103.6% | 91.4% |
| Rows | 2,364 | 4,325 |

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__years_since_debut` | 2,221 |
| `numeric__age` | 2,050 |
| `numeric__height_in_cm` | 1,402 |
| `numeric__minutes_per_appearance` | 1,170 |
| `numeric__minutes_played` | 1,013 |
| `numeric__cards_per_90` | 927 |
| `numeric__appearances` | 896 |
| `numeric__goals_per_90` | 736 |
| `numeric__assists_per_90` | 697 |
| `numeric__seasons_observed` | 582 |

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 60% worse than a random split would suggest. The reported number is the honest one.
- **Weakest measured segment: value band <1M** — MAPE 140% over 1,511 rows.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
