# Methodology

The reasoning behind each decision in the pipeline, moved here so the README
can stay a landing page. Nothing in this file is a summary — it is the working,
with the measurements that produced each choice.

Companion documents: [`DATASET_CARD.md`](DATASET_CARD.md) for what the data is,
[`EXPERIMENT_TRACKING.md`](EXPERIMENT_TRACKING.md) for how runs are recorded,
[`../plans/`](../plans/) for the phase-by-phase audit record.

---

## 1. Label construction

The label is the first valuation recorded **after** the season's evidence is
complete — an as-of join, never an equality merge, because valuations are an
irregular on-change event series rather than a fixed grid.

### The tolerance window

That join used a **120-day** tolerance and discarded 61% of the panel. A season
ending 1 July gives a window closing 29 October, which is before Transfermarkt's
winter revaluation batch exists. Measured across the full panel:

| Tolerance | Player-seasons labelled |
|---|---|
| 120 days | 39.0% |
| 180 days | 74.7% |
| **365 days** | **90.8%** |

Widening cannot leak: `direction="forward"` is unchanged, so every label is
still strictly after the features. It did expose a leak — with a year-wide
window, season *s* can be labelled after season *s+1* has begun, making *s+1*'s
"previous value" not previous. 22 rows in 61,555, and now a seventh leakage
check ([`../src/validation/leakage.py`](../src/validation/leakage.py)) that
fails the build if it recurs.

### The two variants

Two model variants come out of the one table: *performance-only* (every row;
the useful model, for scouting) and *with prior value* (61,522 rows; the
accurate model, for tracking). Shipping only the second would be technically
true and practically useless.

They differ by a row filter and two columns. Performance-only is fitted on the
**54** declared features; with-prior-value adds
`prev_log_market_value_in_eur` and `prev_value_age_days` for **56**, and drops
every row that has no earlier valuation to lag. The lagged value is the target
from an earlier season, which is why it is named with the `prev_` prefix the
leakage detector accepts as proof a copy of the target is deliberate — and why
`prev_value_age_days` exists beside it, since a valuation one season old and
one four years old deserve different weight.

### What the table grew to

| | before Phase 15 | now |
|---|---|---|
| rows | 36,880 | **85,966** |
| players | 17,053 | **24,411** |
| seasons | 2011–2024 | 2011–2024 |
| rows with a prior-season value | 19,827 | **61,522** |
| features | 19 | **54** |
| current-season rows, predictable | 0 | **8,709** |
| leakage findings | 0 | 0 |

---

## 2. The three features that needed care rather than code

**`competition_value_level`** is the honest answer to "how strong is this
league", and the honest answer is the market value of the players in it — which
is the target. Computed from the current season that is textbook target
leakage. It uses a **strictly expanding window**: the level for a competition in
season *s* is the mean of seasons *< s* only, via `shift(1)` then
`expanding()`. Asserted three ways in
[`../tests/unit/test_context.py`](../tests/unit/test_context.py), including that
perturbing the last season cannot change any earlier feature.

**Club strength** comes from `club_games.csv` — actual results — and not from
`clubs.csv`, whose squad value and size are *current* state. Joining today's
squad value to a 2013 row is the same error as joining a contract expiry date,
which this project already bans. `clubs.csv` is the one file of the ten that is
deliberately not downloaded.

It is also joined **as of each row's own date** rather than averaged over the
season, and that is a correction rather than a refinement. A season is named
for its August and runs to July, so a fixture on 20 July belongs to it while
falling after the 1 July boundary most as-of dates sit on — 5.56% of all
fixtures are dated in July. The whole-season mean therefore folded matches
played *after* the row's as-of date into 17.6% of rows, and matches played
after the **label** into 1,049 of them. That is label leakage, and it survived
every existing check because it lived inside a number rather than in a column
name or a date. `club_season_strength` now returns a running record and
`attach_context` cuts it with `merge_asof`; removing the leak cost nothing
measurable (p = 0.49 and p = 0.89). Full workings in
[`../plans/06-final-research-audit.md`](../plans/06-final-research-audit.md).

**Career momentum** is the block that finally gave the performance-only variant
a memory. It knew how established a player was and nothing about what he had
done, so a 24-year-old with 2,800 minutes looked the same whether last season
was 2,900 minutes or 300. These are lagged *features*, never a lagged label,
which is what makes them legal in the variant that withholds the prior
valuation. Pooled over five held-out seasons at three seeds they move
performance-only from €1,976,268 to €1,850,058 (−6.4%, t = 11.30) and leave the
prior-value variant statistically unchanged.

---

## 3. Baselines, and why the temporal split is the headline

Gradient boosting, test-set metrics in EUR, from `scripts/train_baseline.py`.
The **temporal** row is the headline: it is the only split where the model has
never seen the season it is asked about, which is the only situation it will
meet in use.

| Split | performance-only | + prior value |
|---|---|---|
| Random (flattering) | R² 0.805 / MAE €1.63M | R² 0.903 / MAE €1.27M |
| Group by player | R² 0.795 / MAE €1.60M | R² 0.912 / MAE €1.37M |
| **Temporal** | **R² 0.770 / MAE €2.31M** | **R² 0.899 / MAE €1.71M** |

The gap between the random and temporal rows is the point. Reporting the random
number would roughly halve the stated error and answer a question nobody
deploying this will ever ask.

Everything is seeded from one constant; two runs agree exactly, and a test
asserts it. The leakage checks re-run after every split, because splitting is
what creates the chance of a row or a player straddling the boundary.

---

## 4. Model selection

Eleven families (Linear, Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees,
HistGradientBoosting, XGBoost, LightGBM, CatBoost, and a ridge-blended stack of
the three boosters), each searched on expanding-window folds inside the training
seasons, then scored once on the test seasons. LightGBM ships for both variants.

**"Wins" is doing less work than it looks.** The top four families on the
performance-only variant are separated by EUR 44,000 of validation MAE, and the
standard error of that MAE is EUR 60,000. They are the same model as far as this
data can tell. On the prior-value variant the top three explainable families
sit inside EUR 5,600 of one another — stacked 1,651,350, CatBoost 1,655,749,
LightGBM 1,656,909 — against a standard error an order of magnitude larger, so
the one-standard-error rule decides it on deployment footprint and LightGBM
ships.

So selection is not `min()`. Two rules decide what ships:

**It must be explainable.** Every prediction response documents an
`explanation`, the dashboard draws a contribution chart from it, and both model
cards are built from named importances. The stacked blend had the lowest
validation MAE on both variants and exposes no importances at all — its members
have them, the blend does not. HistGradientBoosting is subtler and shipped once
before this rule existed: SHAP works on it, so every prediction looked
explained, while `feature_importances_` has never existed on it and the model
card came out blank. Both are excluded, both still run, and the leaderboard
shows what excluding them costs — which, after the final audit's feature work,
is €10,152 of validation MAE at p = 0.32. The constraint is now free.

**Then the one-standard-error rule** (Breiman, Friedman, Olshen & Stone, 1984):
among families within one standard error of the best, take the cheapest.
XGBoost beat LightGBM by EUR 7,917 — a seventh of one standard error, paired
t = 0.37 — and shipping it would have added 372 MB to the serving image, 291 MB
of that being CUDA libraries a CPU inference path never opens. The tiebreak is
deployment footprint rather than artifact size, because artifact size points the
wrong way: CatBoost serialises to a fifth of LightGBM's file and costs 328 MB of
package.

**The zoo still barely beats the baseline.** Eleven families and a
hyperparameter search buy a few hundredths of R², which is worth stating
plainly rather than burying: the signal in this data is in the features, not in
the estimator. v1.2.0 moved R² from 0.441 to 0.813 by adding features. The
final audit moved it again, to 0.833, by adding thirteen more — and separately
found that the largest remaining estimator-side gain was not a better family at
all but `min_child_samples` and `reg_lambda`, which had never been in the grid.

Each run writes a versioned artifact to `models/` — the fitted preprocessing
and estimator as one object, plus metrics, named feature importance, the full
leaderboard and the seed — with a readable JSON sidecar. A test asserts a
reloaded artifact reproduces its recorded metrics exactly. The contract is
[`EXPERIMENT_TRACKING.md`](EXPERIMENT_TRACKING.md).

---

## 5. Limitations, in full

The README carries a summary of these. This is the working behind each.

### Coverage begins in 2012, and extending it is possible and pointless

`appearances.csv` has no row before 2012-07-03 and `game_lineups.csv` none
before 2013, so no earlier season has performance data. Valuations, though,
reach back to 2000. The final audit built the rows that fact allows —
**40,339 of them, 2003–2011, 10,741 players**, a 66% increase in training data
— from biography and a prior valuation alone. Only **7 of the 54 declared
features exist on such a row** (age, age², height, position, sub-position, foot,
citizenship); the rest have no data to come from. Adding all 40,339 rows moved
pooled held-out MAE by €2,873, or 0.19%: **t = 0.35, p = 0.73**. That experiment
was run against the 41-feature list of the time, where 34 of 41 were missing;
the momentum block only makes such a row emptier, never fuller. A row missing
most of its features teaches the imputer's median, not the mapping. The
limitation is real, and it is now measured rather than asserted. Career-length
features stay left-censored and capped at 10 for the same reason.

### The labels are Transfermarkt's community estimates, not prices anyone paid

`transfers.csv` carries real fees and `--target transfer_fee` will train on
them, but it covers 8.5% of the table against the market value's 83%, and only
players who were actually sold — so it learns what a sold player costs, which
is not what a player is worth. The default target is the appraisal, and the
model reproduces that consensus including wherever it is biased.

Four alternative targets were measured against it rather than reasoned about.
A **delta-value** target is the same model re-parameterised and nominally worse
(p = 0.37). A **within-season percentile** ranks identically (Spearman 0.907 vs
0.909) but costs 50–66% more MAE once converted back to euros, because the
inverse map needs the target season's value distribution and nobody has that at
prediction time. **Career peak** is right-censored on 28.4% of rows in a way
that correlates with the target. None of them is better; the appraisal stays.

### Error grows with value

The target spans four orders of magnitude and the headline MAE is in EUR, so a
mid-table figure conceals much larger absolute misses at the top of the market.
Read the per-band breakdown in `reports/error_analysis.html` before trusting a
number for an expensive player.

### The prediction intervals are wide, and cover slightly less than they say

A gradient booster has no calibrated uncertainty; the interval is measured from
the model's own residual quantiles. Those quantiles now come from the
**validation** season and their coverage is measured on the **test** seasons —
until the final audit both came from the test seasons, which makes a nominal
80% interval cover exactly 80% by construction and is arithmetic rather than
evidence. Measured honestly it reaches about **0.77** against a nominal 0.80
for the performance-only model, and 0.80 for the prior-value one. Every
response now carries `measured_coverage` beside `level`, and the dashboard
shows both.

The shortfall is not closed on purpose. Conformal prediction guarantees
coverage under exchangeability and consecutive football seasons are not
exchangeable; widening the bounds until the number read 0.80 would mean fitting
them to the test set. Conformalised quantile regression, symmetric split
conformal and raw LightGBM quantile regression were all measured and all lose
on Winkler score to the method already here. Wide is still the finding.

### The best model does not ship, and that now costs almost nothing

A stacked blend of the three boosters cannot produce a feature importance or a
SHAP value, and every prediction response documents an explanation, so it is
excluded by `EXPLAINABLE_REQUIRED` in
[`../src/pipelines/tune.py`](../src/pipelines/tune.py). At v1.2.0 that cost
1.92% and 0.33% of validation MAE. On the corrected feature set the blend beats
LightGBM by €10,152 with **p = 0.32** — the two are indistinguishable, because
the career-momentum block gave LightGBM most of what the ensemble was buying.

Knowledge distillation was tested rather than assumed. A LightGBM student
taught on 75% true labels and 25% of the stacked teacher's beat the shipped
model by €22,333 (t = 3.41, p = 0.0006) while staying fully explainable, which
looked like a reason to add a teacher to every training run. It was not: a
LightGBM teaching *itself* gains nothing (p = 0.81), and against a properly
**regularised** LightGBM the student is worth €2,487 at p = 0.72. The ensemble
was standing in for regularisation the search had never been allowed to try,
because `min_child_samples` and `reg_lambda` were not in the grid. They are
now, which is where that €22,333 actually went — four times the grid for one
family, no teacher, and nothing added to the serving image.

### Seasons are August to July, and no single boundary is right for every league

`games.csv` carries each competition's own season label, so this was testable
rather than merely regrettable. The two disagree on 6.27% of fixtures —
Brazilian Série A, the J1 League, Eliteserien, Allsvenskan, the K-League, MLS
and every European qualifying round played in July — and 19.6% of
player-seasons span the boundary, though only 2.02% of minutes are misfiled.
The harm is real: rows with more than 5% of their minutes misfiled carry
0.04–0.12 more median relative error *within every value band*, and that
survives controlling for the population those July fixtures select.

A full alternative table indexed on the declared season was built anyway. It
first appeared to be a 10% improvement, and it was an artefact: anchoring the
as-of date on each competition's last fixture moved it earlier on 93% of rows
and cut the median label horizon from 141 days to 23, so the model was
forecasting three weeks ahead instead of five months. Rebuilt with the horizon
held fixed, the gain is null for the scouting model (p = 0.77, 0.06, 0.25) and
significantly *worse* for the prior-value one (−€68,996, p = 0.0005), on 6.5%
fewer rows.

The reason is the limitation itself: the declared index groups July qualifiers
correctly but then leaves the 1 July label anchor seven months adrift of a
Brazilian season ending in December. This panel has no anchor that is right for
every calendar, and the fixed-July one is wrong on fewer of the rows that
matter. Measured, not assumed —
[`../plans/06-final-research-audit.md`](../plans/06-final-research-audit.md) §5.
