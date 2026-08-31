# Model card — with_prior_value

Generated 2026-08-30T23:43:32.905750+00:00 from `with_prior_value__lightgbm`.
This file is written from the artifact, so it cannot describe a model that
is no longer the one on disk.

## What it does

Forecast how a player's *already known* market value will move. This is the tracking model. It is substantially more accurate and substantially less interesting: most of its skill comes from the prior valuation, so it cannot tell you the market is wrong.

## Not what it is for

- Setting or negotiating an actual transfer fee. Market value and transfer fee are different quantities; this model is trained on the former.
- Players outside the covered competitions, or below the appearance volume the training data contains.
- Any individual decision about a person's employment or compensation without a human reviewing the explanation alongside the number.

## Model

| Family | `lightgbm` |
|---|---|
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.05, 'model__regressor__num_leaves': 63}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 43 |
| Seed | 42 |

## Data and split

- Rows: 61,522
- Split: temporal — train ≤2021, validation 2022, test ≥2023
- Source: Kaggle `davidcariboo/player-scores` (CC0). Transfermarkt is
  never scraped; its terms prohibit both the method and this purpose.

## Measured performance

Test seasons, in EUR. These are held-out seasons the model never saw.

| Metric | Validation | Test |
|---|---|---|
| MAE | €1,668,757 | €1,661,311 |
| RMSE | €4,282,374 | €4,077,559 |
| R² | 0.890 | 0.914 |
| MAPE | 29.5% | 30.1% |
| Rows | 5,101 | 10,169 |

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__age` | 2,097 |
| `numeric__prev_value_age_days` | 1,697 |
| `numeric__competition_value_level` | 1,531 |
| `numeric__prev_log_market_value_in_eur` | 1,398 |
| `numeric__competition_tier_rank` | 825 |
| `numeric__club_goal_difference_per_game` | 758 |
| `numeric__years_since_debut` | 731 |
| `numeric__club_points_per_game` | 730 |
| `numeric__minutes_played` | 659 |
| `numeric__club_league_position` | 620 |

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 60% worse than a random split would suggest. The reported number is the honest one.
- **This variant is anchored to the previous valuation.** It will track the market rather than challenge it, and it cannot produce a prediction at all for a player with no valuation history.
- **Weakest measured segment: value band <1M** — MAPE 34% over 3,493 rows.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
