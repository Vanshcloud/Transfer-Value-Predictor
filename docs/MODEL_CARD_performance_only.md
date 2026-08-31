# Model card — performance_only

Generated 2026-08-30T23:31:52.501716+00:00 from `performance_only__lightgbm`.
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
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.1, 'model__regressor__num_leaves': 63}` |
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
| MAE | €2,063,630 | €2,205,618 |
| RMSE | €5,271,584 | €5,419,571 |
| R² | 0.798 | 0.813 |
| MAPE | 51.8% | 58.0% |
| Rows | 6,573 | 13,486 |

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__competition_value_level` | 1,446 |
| `numeric__age` | 1,298 |
| `numeric__years_since_debut` | 1,118 |
| `numeric__club_goal_difference_per_game` | 1,056 |
| `numeric__club_league_position` | 1,048 |
| `numeric__club_points_per_game` | 938 |
| `numeric__competition_tier_rank` | 796 |
| `numeric__squad_match_share` | 783 |
| `numeric__height_in_cm` | 686 |
| `numeric__minutes_played` | 618 |

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 60% worse than a random split would suggest. The reported number is the honest one.
- **Weakest measured segment: value band <1M** — MAPE 77% over 5,463 rows.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
