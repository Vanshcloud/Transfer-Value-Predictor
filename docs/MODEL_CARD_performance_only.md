# Model card — performance_only

Generated 2026-10-18T14:35:25.296453+00:00 from `performance_only__lightgbm`.
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
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.1, 'model__regressor__num_leaves': 63, 'model__regressor__min_child_samples': 120, 'model__regressor__reg_lambda': 20.0}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 54 |
| Seed | 42 |

## Data and split

- Rows: 85,966
- Split: temporal — train ≤2021, validation 2022, test ≥2023
- Source: Kaggle `davidcariboo/player-scores`, a mirror of Transfermarkt
  data its publisher dedicates as CC0. Transfermarkt is never scraped, but
  its terms also prohibit training machine learning on the content, which
  this model is; the usage rights are uncertain and not cleared
  ([Data rights](DATASET_CARD.md#data-rights)).

## Measured performance

Test seasons, in EUR. The model was not fitted on them, but they were
inspected during development, so these figures are **retrospective**
rather than independent ([EVALUATION.md](EVALUATION.md)).

| Metric | Validation | Test |
|---|---|---|
| MAE | €1,892,889 | €2,057,285 |
| RMSE | €4,737,305 | €4,997,733 |
| R² | 0.837 | 0.841 |
| MAPE | 47.3% | 52.3% |
| Rows | 6,573 | 13,486 |

## Prediction intervals

| Nominal level (what the interval targets) | 80% |
|---|---|
| Residual quantile level used | 82.5% |
| Simulated coverage before widening (pre-test seasons) | 77.7% |
| Simulated coverage after widening (pre-test seasons) | 80.3% |
| Quantiles fitted on | season 2022, 6,573 rows |
| **Observed coverage, test seasons** | **79.1%** of 13,486 rows |
| Median width | €2,233,816 |

| Test subgroup | Observed coverage | Rows |
|---|---|---|
| season 2023 | 81.2% | 5,726 |
| season 2024 | 77.5% | 7,760 |
| predicted 1M-5M | 79.7% | 4,409 |
| predicted 20M-50M | 89.2% | 593 |
| predicted 5M-20M | 83.1% | 2,020 |
| predicted <1M | 76.5% | 6,342 |
| predicted >50M | 77.0% | 122 |

The quantiles come from the validation season's residuals. They are
widened to the level the pre-test simulation needed to hold the nominal
level one and two seasons ahead — a model fitted before each of five
origin seasons, calibrated on it and scored on the next two, all before
the test seasons. Coverage is then measured on the test seasons, which
neither the model nor the interval saw.

That test figure is **retrospective**, not independent: the widening was
introduced after the unwidened interval's test coverage had been seen.
Coverage is a frequency over a group of predictions. It is not the
probability that any one player's value lies inside their interval, and
small subgroups (the >50M band) are measured on few rows.

## What it relies on

| Feature | Importance |
|---|---|
| `numeric__competition_value_level` | 1,234 |
| `numeric__age` | 1,134 |
| `numeric__years_since_debut` | 881 |
| `numeric__club_goal_difference_per_game` | 868 |
| `numeric__competition_tier_rank` | 848 |
| `numeric__club_league_position` | 837 |
| `numeric__age_squared` | 790 |
| `numeric__club_points_per_game` | 733 |
| `numeric__squad_match_share` | 647 |
| `numeric__minutes_played` | 486 |

## Assumptions

- **The future resembles the recent past.** Training seasons are weighted toward recent ones, but a structural break — a new financial-fair-play regime, another closed-doors season — is not something the model can anticipate. Season 2020 is already such a break in this data.
- **A player's competition is a proxy for the quality he faced.** The model never sees an opponent. It sees the league's historical value level and his club's results, which is an average standing in for a specific.
- **Minutes are earned, not assigned.** Availability is read as a signal of quality. A player kept out by injury and one kept out by his manager look the same here.
- **The valuation being predicted is set within a year of the as-of date.** It is the first one published in that window, so the horizon varies from row to row and the model is not told it: a prediction is the next published valuation, not the value on a fixed date. Widen the window and the question changes; it is recorded per row as `label_horizon_days`.

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 45% worse than a split grouped by player would suggest. The reported number is the honest one.
- **Weakest measured segment: value band <1M** — MAPE 69% over 5,463 rows.

## Failure modes

- **Players with no history.** A first covered season has no lagged features and the model falls back on biography and current output. The `career_stage` breakdown in the error report isolates exactly these rows.
- **Sudden reputational moves.** A transfer saga, a tournament breakout or a long-term injury repriced the player faster than any season-level feature can register. The model is smooth; the market is not.
- **The very top of the market.** Above EUR 50M there are few comparable seasons and the absolute error is largest. Relative error is smallest there, which is a different statement and easy to confuse.
- **Leagues thinly represented in the panel.** Competition strength is an expanding historical mean, so a league with little history gets a null and the imputer's median instead of a level.
- **Anything outside the covered competitions.** No row exists, so no prediction is served rather than a guess being manufactured.

## Fairness

The model is not audited against protected attributes as a classifier would be — there is no favourable outcome to allocate, only an estimate that is more or less accurate. What matters here is whether it is *reliably* accurate across groups, so the measured spread is reported instead of a fairness score.

Worst and best competitions by MAPE, over segments with enough rows to mean anything:

| Competition | Rows | MAE | MAPE |
|---|---|---|---|
| `CLQ` | 314 | EUR 1,186,429 | 103% |
| `CDR` | 124 | EUR 860,765 | 99% |
| `DKP` | 85 | EUR 317,741 | 96% |
| `RU1` | 662 | EUR 987,416 | 37% |
| `IT1` | 917 | EUR 3,405,434 | 36% |
| `GR1` | 583 | EUR 755,927 | 36% |

A wide spread here means the model serves some leagues better than others, which is a property of how much history the panel holds for each. Read it before quoting one number for every competition.

## Ethical considerations

- **This is decision support, not a decision.** Every response carries an explanation and an interval so a human can disagree with it. A valuation used to set a person's wages or transfer terms without that human is a misuse of the model.
- **The labels encode a community's opinion of people.** Transfermarkt valuations are crowd estimates, and any bias in that crowd — toward visible leagues, particular nationalities, particular styles — is reproduced faithfully. The model cannot correct a bias it is trained to imitate.
- **Nationality is a feature.** `country_of_citizenship` improves accuracy and is a protected attribute. It is retained because removing it does not remove the information (league and club are proxies) and does hide it. Stated plainly so the choice is reviewable rather than invisible.
- **No personal data beyond the public record.** Biography, appearances and public valuations only. Nothing here is scraped, and nothing about a player's private life is used or inferred.

## Leakage controls

- Every feature is observed at or before the label date, enforced by
  construction and asserted on all rows.
- Current-state columns (`contract_expiration_date`, the
  `current_club_*` family) never enter the feature matrix.
- The prior valuation, where used, is explicitly lagged and named so.
- Splits are re-checked for row and player overlap after every split.
- Club form is joined as of the row's own date, so a match played after
  the label was set cannot reach the features through an aggregate.
