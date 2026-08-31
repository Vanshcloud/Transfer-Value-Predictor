# Model card — performance_only

Generated 2026-08-31T19:07:45.881722+00:00 from `performance_only__lightgbm`.
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
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.1, 'model__regressor__num_leaves': 63, 'model__regressor__min_child_samples': 120, 'model__regressor__reg_lambda': 0.0}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 54 |
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
| MAE | €1,938,817 | €2,068,081 |
| RMSE | €4,978,845 | €4,995,721 |
| R² | 0.820 | 0.841 |
| MAPE | 47.6% | 52.2% |
| Rows | 6,573 | 13,486 |

## Prediction intervals

| Nominal level | 80% |
|---|---|
| Measured coverage | 76.9% |
| Median width | €2,052,897 |
| Measured on | 13,486 test rows |

Quantiles are taken from the validation season and the coverage above
is measured on the test seasons, which neither the model nor the
interval has seen. Measured coverage below the nominal level is the
cost of predicting forward across seasons that are not exchangeable.

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__competition_value_level` | 1,258 |
| `numeric__age` | 1,185 |
| `numeric__years_since_debut` | 902 |
| `numeric__club_goal_difference_per_game` | 815 |
| `numeric__club_league_position` | 812 |
| `numeric__club_points_per_game` | 760 |
| `numeric__competition_tier_rank` | 734 |
| `numeric__age_squared` | 696 |
| `numeric__squad_match_share` | 665 |
| `numeric__prev_club_points_per_game` | 507 |

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 60% worse than a random split would suggest. The reported number is the honest one.
- **Weakest measured segment: value band <1M** — MAPE 69% over 5,463 rows.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
- Club form is joined as of the row's own date, so a match played after
  the label was set cannot reach the features through an aggregate.
