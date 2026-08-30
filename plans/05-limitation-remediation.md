# Phase 15 — attacking the dataset limitations

The Phase 14 audit closed every defect it found and then listed seven things it
called limitations of the data rather than of the code. This phase treats that
distinction as a hypothesis and tests it.

Six of the seven turned out to be limitations of the *pipeline*. One is real.

## The finding that reframed everything

`configs/config.yaml` downloaded three files:

```yaml
files:
  - players.csv
  - player_valuations.csv
  - appearances.csv
```

The Kaggle dataset ships **ten**. The seven that were never fetched carry
competition identity, club results, starting line-ups and transfer fees — which
is to say, four of the seven "unavoidable" limitations were sitting unopened in
the same download the project already had credentials for.

| File | Size | What it carries |
|---|---|---|
| `competitions.csv` | 11 KB | competition type, confederation, country |
| `games.csv` | 25 MB | official season label, competition, attendance |
| `club_games.csv` | 11 MB | per-club results — points, goal difference, position |
| `game_lineups.csv` | 352 MB | **starting eleven vs substitute**, position, captaincy |
| `transfers.csv` | 15 MB | **actual transfer fees** |
| `game_events.csv` | 157 MB | goals, cards, substitutions with minutes |
| `clubs.csv` | 184 KB | current squad value, size, age — *deliberately unused* |

`clubs.csv` is downloaded by nobody here on purpose. Its interesting columns
are all *current* state, exactly like the banned columns in `players.csv`:
joining today's squad value to a 2013 row states a fact from 2026. Club
strength is derived from `club_games.csv` instead, which is dated results.

## L1 — 64.3% of player-seasons discarded

**Solved: 39.0% labelled → 83.2%.**

The label was the first valuation within **120 days** of the as-of date. A
season ending 1 July gives a window closing 29 October — which is *before*
Transfermarkt's winter revaluation batch exists. Measured across the panel:

| Tolerance | Labelled | Median horizon |
|---|---|---|
| 120 days | 39.0% (36,902) | 53 days |
| 180 days | 74.7% (70,735) | 114 days |
| **365 days** | **90.8% (86,016)** | 141 days |

Widening cannot leak. `direction="forward"` is unchanged, so every label is
still strictly after the features. What changes is how far ahead it sits.

An anchor-based alternative was measured and rejected: taking the valuation *in
force* at a fixed post-season date reaches 98% coverage, but only 20–40% of
those labels are genuine post-season revaluations — the rest are stale
pre-season values carried forward, which do not reflect the season at all.
Coverage bought by making the label mean less is not coverage.

`label_horizon_days` was added as a feature to let the model condition on how
far ahead it was predicting. It earned nothing — test R² 0.760 with it against
0.762 without — so it was demoted to a diagnostic column. A feature that moves
the third decimal the wrong way is not worth the API it would add.

**It also introduced a leak, which is the part worth reading.** A year-wide
window means season *s* can be labelled *after* season *s+1* has begun, at
which point *s+1*'s "previous value" is not previous. 22 rows in 61,555 —
0.03%, far too small to show in any metric. `add_career_history` now requires
the prior label to strictly precede the as-of date, and
`check_lagged_values_precede_features` is a seventh leakage check that fails
the build if it ever recurs.

## L2 — five performance statistics

**Solved: 19 features → 41.**

`appearances.csv` gives matches, minutes, goals, assists and cards. The other
files give role, availability, consistency and trajectory:

- **Role** — `starts`, `substitute_appearances`, `start_share`, `captain_share`,
  `positions_played`, from `game_lineups.csv`. Two thousand minutes across 25
  starts is a different player from 2,000 across 40 appearances, and total
  minutes cannot tell them apart.
- **Availability** — `squad_match_share`: matches played over matches the club
  actually played. Thirty appearances means something different in a 34-game
  league than in a 60-game European campaign.
- **Consistency** — `full_match_share`, `minutes_variability`,
  `scoring_match_share`, `months_active`.
- **Trajectory** — `second_half_goal_share`. Fifteen goals after Christmas is
  priced differently from fifteen before it, and the label is set in the summer
  that follows.

## L3 — no league, club or competition strength

**Solved.** Seven context features, and one of them needed real care.

`club_points_per_game`, `club_goal_difference_per_game` and
`club_league_position` come from results *inside* the season being described.
That is safe without argument: the as-of date is the later of the season
boundary and the player's last appearance, so every result is already observed.

`competition_value_level` is the dangerous one. The natural measure of league
strength is the market value of the players in it — and that is the target.
Computed from the current season it would be textbook target leakage.

It is computed with a **strictly expanding window**: the level for competition
*c* in season *s* uses only seasons *< s*. `shift(1)` then `expanding().mean()`,
so a season never contributes to its own feature. Asserted three ways —
a synthetic competition whose values explode in one season must not see the
explosion in that season's feature; perturbing the last season must not change
any earlier feature; and on the real panel the feature correlates more tightly
with the *previous* season's level (0.929) than with its own (0.919).

## L4 — coverage starts in 2012

**Not solved. It is a real limit of this dataset.**

`games.csv` reaches back to 2006-06-09 and `game_events.csv` to the same date,
which looks like five extra seasons. It is not:

```
game_events before 2012-07-03 :  2,470 rows, 190 games, 943 players
game_lineups earliest date    :  2013-07-02
```

Those 190 games are international tournaments, not league football. There is no
per-player league record before 2012-07-03 to aggregate, and no line-up data
before 2013 to say who was even on the pitch. Reconstructing minutes from
substitution events is impossible in principle: a player who plays the full
match generates no event at all.

Extending coverage needs a different data source, not better use of this one.

## L5 — Transfermarkt values are community estimates

**Implemented as an optional mode; the default is unchanged, with reasons.**

`transfers.csv` carries real fees. Measured:

```
transfer rows                    175,165
with a fee above zero             17,554   (10.0%)
joinable to a player-season        7,326   (8.5% of the table)
corr(log market value, log fee)    0.863
median fee / market value           0.93
```

Two problems, one fatal. **Coverage**: 8.5% against 83%. **Selection**: a fee
exists only where a transfer happened, and transfers are not random — players
move when a club wants to sell, when a contract runs down, when form spikes.
Training on fees learns "what does a sold player cost", which is not "what is
this player worth", because most players in any season are not sold.

So `--target transfer_fee` exists for anyone who wants to model the price
rather than the appraisal, and the market value stays the default.

## L6 — the freshest season is a year old

**Solved: 8,709 rows for the season in progress are now predictable.**

A row could not exist until a valuation existed to label it, and that valuation
appears only after the season ends. That is correct for *training* and was
being applied to *prediction*, where it is not needed at all: the features are
complete the moment the matches are played.

`build_current_season_table` produces exactly those rows through the same
enrichment code the training table uses — not a copy, the same function, so a
prediction row cannot be assembled differently from the rows the model learnt
from. They carry `has_label = False`, are never returned to training, and take
their prior value from the raw valuations by a strictly backward as-of join
(median staleness 34 days, against 242 in the training table).

A half-played season is priced as less evidence, not refused: `appearances`,
`squad_match_share` and `months_active` all fall, which is the honest signal.

## L7 — imbalanced seasons

**Mostly solved as a side effect, then measured properly.**

Widening the label window fixed most of it on its own: 310–5,674 rows per
season became 1,093–7,760, an 18.3× spread down to 7.1×.

Five weighting schemes were then fitted on the training seasons and scored on
the **validation** season, with the test seasons untouched until one was chosen:

```
none                validation MAE EUR 2,169,834
inverse-frequency   validation MAE EUR 2,237,664   worst
sqrt inverse-freq   validation MAE EUR 2,164,614
recency             validation MAE EUR 2,126,386   best
inverse x recency   validation MAE EUR 2,147,905
```

Inverse-frequency balancing — the obvious answer to "some seasons are small" —
was the worst of the five. The imbalance is not what hurts. Football's market
inflates and squads turn over, so a 2011 row is a weaker guide to 2024 than a
2020 row regardless of how many of each there are; up-weighting scarce old
seasons makes the model more like the past, which is the opposite of what a
temporal split rewards.

## The model that won, and why it does not ship

Eleven families now, adding Extra Trees and a ridge-blended stack of the three
boosters. The **stack won both variants** on validation.

It does not ship. `docs/API_CONTRACT.md` documents an `explanation` object on
every prediction; the dashboard draws a contribution chart from it; both model
cards are built from named feature importances. A `StackingRegressor` exposes
neither `feature_importances_` nor `coef_` — its members each have them, but
the blend does not, and averaging the members would describe a model that is
not the one predicting. SHAP has the same problem: TreeExplainer cannot walk a
blend, and KernelExplainer costs seconds per request against milliseconds.

The cost of refusing it is measured and logged rather than hidden:

```
performance_only   stacked 2,024,017  vs  best explainable 2,063,630   1.92%
with_prior_value   stacked 1,663,283  vs  best explainable 1,668,757   0.33%
```

Under 2% of validation MAE, against every prediction losing its explanation and
the serving image regaining the 700 MB of xgboost and catboost that
`requirements-serve.txt` exists to leave out. The stack stays in the zoo so the
leaderboard shows what the constraint costs, and
`src.pipelines.tune.EXPLAINABLE_REQUIRED` is the one line to flip for anyone
who wants the 2% back.
