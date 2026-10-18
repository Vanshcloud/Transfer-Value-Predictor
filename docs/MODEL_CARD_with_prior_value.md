# Model card — with_prior_value

Generated 2026-10-18T15:08:34.859418+00:00 from `with_prior_value__lightgbm`.
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
| Hyperparameters | `{'model__regressor__n_estimators': 300, 'model__regressor__learning_rate': 0.05, 'model__regressor__num_leaves': 63, 'model__regressor__min_child_samples': 20, 'model__regressor__reg_lambda': 0.0}` |
| Target | `market_value_in_eur`, trained on `log1p`, reported in EUR |
| Features | 56 |
| Seed | 42 |

## Data and split

- Rows: 84,626
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
| MAE | €900,232 | €908,625 |
| RMSE | €2,360,953 | €2,406,492 |
| R² | 0.960 | 0.963 |
| MAPE | 21.2% | 20.8% |
| Rows | 6,493 | 13,315 |

## Prediction intervals

| Nominal level (what the interval targets) | 80% |
|---|---|
| Residual quantile level used | 80.0% |
| Simulated coverage before widening (pre-test seasons) | 80.2% |
| Simulated coverage after widening (pre-test seasons) | 80.2% |
| Quantiles fitted on | season 2022, 6,493 rows |
| **Observed coverage, test seasons** | **80.1%** of 13,315 rows |
| Median width | €1,007,299 |

| Test subgroup | Observed coverage | Rows |
|---|---|---|
| season 2023 | 80.2% | 5,666 |
| season 2024 | 80.1% | 7,649 |
| predicted 1M-5M | 81.3% | 4,379 |
| predicted 20M-50M | 84.1% | 790 |
| predicted 5M-20M | 82.0% | 2,267 |
| predicted <1M | 78.2% | 5,677 |
| predicted >50M | 72.3% | 202 |

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
| `numeric__age` | 1,851 |
| `numeric__prev_log_market_value_in_eur` | 1,733 |
| `numeric__prev_value_age_days` | 1,180 |
| `numeric__years_since_debut` | 874 |
| `numeric__competition_value_level` | 849 |
| `numeric__club_points_per_game` | 729 |
| `numeric__competition_tier_rank` | 693 |
| `numeric__club_goal_difference_per_game` | 682 |
| `numeric__minutes_played` | 566 |
| `numeric__squad_match_share` | 562 |

## Assumptions

- **The future resembles the recent past.** Training seasons are weighted toward recent ones, but a structural break — a new financial-fair-play regime, another closed-doors season — is not something the model can anticipate. Season 2020 is already such a break in this data.
- **A player's competition is a proxy for the quality he faced.** The model never sees an opponent. It sees the league's historical value level and his club's results, which is an average standing in for a specific.
- **Minutes are earned, not assigned.** Availability is read as a signal of quality. A player kept out by injury and one kept out by his manager look the same here.
- **The valuation being predicted is set within a year of the as-of date.** Widen that horizon and the question changes; the label window is a modelling choice, recorded per row as `label_horizon_days`.

## Limitations

- **Market value is an estimate, not a fact.** The labels come from Transfermarkt's community-maintained valuations. The model reproduces that consensus, including wherever it is biased — it does not independently observe what anyone would pay.
- **Error grows with value.** Errors are reported in EUR, and the target spans four orders of magnitude, so a mid-table MAE conceals much larger absolute misses at the top of the market. Read the per-band breakdown before trusting a number for an expensive player.
- **Coverage begins in 2012.** Appearance data starts 2012-07-03, so career-length features are left-censored and capped; a player whose career began earlier looks younger in career terms than they are.
- **Seasons are August to July.** Leagues on a spring-autumn calendar are split across that boundary and are represented less faithfully.
- **The model has never seen the season it is asked about.** That is deliberate, and it is why the reported error is roughly 45% worse than a split grouped by player would suggest. The reported number is the honest one.
- **This variant is anchored to the previous valuation.** It will track the market rather than challenge it, and it cannot produce a prediction at all for a player with no valuation history.
- **Weakest measured segment: value band <1M** — MAPE 24% over 5,302 rows.

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
| `SFA` | 40 | EUR 84,754 | 39% |
| `DKP` | 74 | EUR 151,961 | 32% |
| `CL` | 109 | EUR 1,050,828 | 29% |
| `GRP` | 46 | EUR 59,997 | 15% |
| `GB1` | 905 | EUR 2,742,432 | 14% |
| `GR1` | 583 | EUR 233,772 | 13% |

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
