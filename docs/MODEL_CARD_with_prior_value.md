# Model card — with_prior_value

Generated 2026-08-31T19:26:16.883404+00:00 from `with_prior_value__lightgbm`.
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
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.05, 'model__regressor__num_leaves': 63, 'model__regressor__min_child_samples': 20, 'model__regressor__reg_lambda': 20.0}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 56 |
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
| MAE | €1,656,909 | €1,637,639 |
| RMSE | €4,261,596 | €4,085,037 |
| R² | 0.891 | 0.914 |
| MAPE | 29.6% | 30.3% |
| Rows | 5,101 | 10,169 |

## Prediction intervals

| Nominal level | 80% |
|---|---|
| Measured coverage | 79.3% |
| Median width | €1,862,759 |
| Measured on | 10,169 test rows |

Quantiles are taken from the validation season and the coverage above
is measured on the test seasons, which neither the model nor the
interval has seen. Measured coverage below the nominal level is the
cost of predicting forward across seasons that are not exchangeable.

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__age` | 1,885 |
| `numeric__prev_value_age_days` | 1,412 |
| `numeric__competition_value_level` | 1,151 |
| `numeric__prev_log_market_value_in_eur` | 1,141 |
| `numeric__competition_tier_rank` | 655 |
| `numeric__club_goal_difference_per_game` | 650 |
| `numeric__club_points_per_game` | 646 |
| `numeric__delta_competition_value_level` | 612 |
| `numeric__delta_minutes_played` | 539 |
| `numeric__years_since_debut` | 514 |

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 60% worse than a random split would suggest. The reported number is the honest one.
- **This variant is anchored to the previous valuation.** It will track the market rather than challenge it, and it cannot produce a prediction at all for a player with no valuation history.
- **Weakest measured segment: value band <1M** — MAPE 35% over 3,493 rows.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
- Club form is joined as of the row's own date, so a match played after
  the label was set cannot reach the features through an aggregate.
