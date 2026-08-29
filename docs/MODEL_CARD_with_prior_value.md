# Model card — with_prior_value

Generated 2026-08-29T09:10:44.583655+00:00 from `with_prior_value__lightgbm`.
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
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.05, 'model__regressor__num_leaves': 31}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 21 |
| Seed | 42 |

## Data and split

- Rows: 19,827
- Split: temporal — train ≤2021, validation 2022, test ≥2023
- Source: Kaggle `davidcariboo/player-scores` (CC0). Transfermarkt is
  never scraped; its terms prohibit both the method and this purpose.

## Measured performance

Test seasons, in EUR. These are held-out seasons the model never saw.

| Metric | Validation | Test |
|---|---|---|
| MAE | €2,988,615 | €3,710,897 |
| RMSE | €7,153,067 | €8,208,767 |
| R² | 0.720 | 0.775 |
| MAPE | 48.0% | 48.7% |
| Rows | 1,569 | 2,455 |

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__prev_value_age_days` | 1,370 |
| `numeric__age` | 1,182 |
| `numeric__prev_log_market_value_in_eur` | 1,035 |
| `numeric__years_since_debut` | 677 |
| `numeric__minutes_per_appearance` | 542 |
| `numeric__minutes_played` | 521 |
| `numeric__appearances` | 472 |
| `numeric__cards_per_90` | 367 |
| `numeric__goals_per_90` | 361 |
| `numeric__height_in_cm` | 298 |

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 60% worse than a random split would suggest. The reported number is the honest one.
- **This variant is anchored to the previous valuation.** It will track the market rather than challenge it, and it cannot produce a prediction at all for a player with no valuation history.
- **Weakest measured segment: value band <1M** — MAPE 59% over 628 rows.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
