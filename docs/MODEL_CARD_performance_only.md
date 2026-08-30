# Model card — performance_only

Generated 2026-08-30T22:12:30.452706+00:00 from `performance_only__xgboost`.
This file is written from the artifact, so it cannot describe a model that
is no longer the one on disk.

## What it does

Estimate a player's market value from on-pitch performance and biography alone, with no prior valuation supplied. This is the scouting and undervaluation model: it is the one that can express an opinion that differs from the market, because it has never been told what the market thinks.

## Not what it is for

- Setting or negotiating an actual transfer fee. Market value and transfer fee are different quantities; this model is trained on the former.
- Players outside the covered competitions, or below the appearance volume the training data contains.
- Any individual decision about a person's employment or compensation without a human reviewing the explanation alongside the number.

## Model

| Family | `xgboost` |
|---|---|
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.1, 'model__regressor__max_depth': 6}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 41 |
| Seed | 42 |

## Data and split

- Rows: 85,966
- Split: temporal — train ≤2021, validation 2022, test ≥2023
- Source: Kaggle `davidcariboo/player-scores` (CC0). Transfermarkt is
  never scraped; its terms prohibit both the method and this purpose.

## Measured performance

Test seasons, in EUR. These are held-out seasons the model never saw.

| Metric | Validation | Test |
|---|---|---|
| MAE | €2,055,713 | €2,241,882 |
| RMSE | €5,169,479 | €5,441,080 |
| R² | 0.806 | 0.811 |
| MAPE | 51.3% | 59.3% |
| Rows | 6,573 | 13,486 |

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__appearances` | 0.1104 |
| `numeric__goal_contributions` | 0.1047 |
| `numeric__competition_value_level` | 0.09959 |
| `numeric__competitions_played` | 0.08972 |
| `numeric__continental_minutes_share` | 0.04911 |
| `numeric__club_goal_difference_per_game` | 0.03673 |
| `numeric__minutes_played` | 0.02731 |
| `numeric__club_points_per_game` | 0.02105 |
| `categorical__country_of_citizenship_Ukraine` | 0.01909 |
| `categorical__country_of_citizenship_Turkey` | 0.01882 |

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 60% worse than a random split would suggest. The reported number is the honest one.
- **Weakest measured segment: value band <1M** — MAPE 79% over 5,463 rows.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
