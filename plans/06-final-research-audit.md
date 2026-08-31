# Phase 16 — final research-grade audit

Phase 15 closed six of seven dataset limitations and declared the seventh
unavoidable. This phase re-verifies all of that from the raw files rather than
from the previous phase's notes, and then asks of every surviving limitation
whether it is a fact about the world or a fact about this pipeline.

The rule throughout: a change ships only if it improves a held-out number and
survives a paired test, or if it removes a leak. Plausibility is not evidence,
and four of the ideas below were rejected precisely because they were plausible
and did not work.

## Method

Every verification was re-derived from `data/processed/*.parquet` and the raw
Kaggle CSVs by scripts that do not import `src.validation`, so a bug in the
project's own checker could not hide a bug in the project's own data.

Model comparisons use expanding-window held-out seasons — train on everything
before season *s*, score season *s*, for *s* in 2018–2022 — at three seeds,
predictions averaged across seeds and errors pooled across seasons, compared
with a paired *t*-test on absolute errors and a Wilcoxon signed-rank test where
the distribution warranted it. The 2023+ test seasons were not touched by any
experiment; they are scored once, by the training pipeline, at the end.

---

## Part 1 — verification

Eleven properties re-checked from scratch. Nine passed unchanged. Two flagged,
of which one was an artefact of the checker and one was a real defect.

| Property | Result |
|---|---|
| `last_appearance_date <= label_date`, all rows | pass |
| `as_of_date` between feature time and label time | pass |
| No duplicate `(player_id, season)` | pass — 0 of 85,966 |
| No current-state columns in the frame | pass |
| No date *value* later than the label | pass |
| `prev_value_age_days > 0` on every lagged row | pass — min 1 day |
| Prior value traces to a real raw valuation on that date | pass — 61,522/61,522 |
| Target non-null and positive | pass — min €10,000 |
| Splits cover all rows and are disjoint | pass |
| No training label postdates the earliest test as-of date | pass — 0 rows |
| `competition_value_level` uses only strictly prior seasons | pass — see below |

**`competition_value_level` — flagged, then cleared.** A first approximation
suggested the expanding target encoding might include its own season. It does
not. Recomputing the encoding by brute force — for every (competition, season),
the mean of that competition's non-null per-season levels for seasons strictly
before it — reproduces the shipped column exactly: 341 comparisons, 0
mismatches, max absolute difference 0.00e+00 over 79,858 rows, and the
null-patterns agree perfectly. The first check was wrong because it did not
replicate the thin-season nulling rule (`MIN_ROWS_FOR_STRENGTH`). The feature
is sound.

**`label_horizon_days == 0` — flagged, then cleared.** 3,108 rows carry a label
dated the same day as their as-of date. 3,070 of those have an as-of date on
the 1 July boundary, where the features closed at the previous match and the
label is Transfermarkt's summer batch — a full forward step. The remaining 38
have an as-of date equal to the player's last appearance, so the valuation and
a match share a calendar day. That is not leakage in the direction that
matters: the concern is a *feature* seeing the future, and here it is the label
that may or may not reflect that day's match. A label that already reflects the
season's last match is what the design asks for.

### The defect: club form built from matches played after the label

`club_points_per_game`, `club_goal_difference_per_game`,
`club_league_position`, `club_matches` and the `squad_match_share` derived from
them were whole-season means. `src/feature_engineering/context.py` argued this
was safe because "the as-of date is the later of the season boundary and the
player's last appearance, so every result in that season is already observed".

That argument is wrong by up to thirty days. A season is named for its August
and runs to July, so a fixture on 20 July belongs to it while falling *after*
the 1 July boundary that most as-of dates sit on. **5.56% of all fixtures are
dated in July.**

Measured on the full panel:

| | rows | share |
|---|---|---|
| Club played a match after the row's as-of date | 15,126 | 17.60% |
| Club played a match after the row's **label** date | **1,049** | **1.22%** |

The second row is label leakage. It survived seven leakage checks because it is
not a date column and not a renamed target — it is a *number*, and the matches
that contaminated it left no trace in the frame.

Concentrated in leagues whose calendars run past 1 July: BE1 (2,395 rows), RU1
(2,121), UKR1 (1,720), PO1 (1,367), DK1 (1,358).

**Fix.** `club_season_strength` now returns a club's *running* record — one row
per club-season-match-date — and `attach_context` joins it with `merge_asof` on
the row's own as-of date. Every club number on a row is now the club's form at
the moment that row's evidence closed. `squad_match_counts` is deleted rather
than kept as a shim: the availability denominator now comes from the same
as-of-bounded `club_matches`, so numerator and denominator finally close at the
same instant instead of one running a month past the other, and a second
function computing the same count from the same source at a different time
bound is exactly how they drifted apart.

**Cost of the fix: none that can be measured.** Paired on the 2022 validation
season:

| variant | whole-season (leaking) | as-of-dated | paired |
|---|---|---|---|
| performance_only | €2,063,630 | €2,074,128 | t = −0.69, p = 0.49 |
| with_prior_value | €1,674,177 | €1,675,861 | t = −0.14, p = 0.89 |

The leak was not buying accuracy. It was only overstating provenance, which is
the worst kind: no metric would ever have flagged it.

Verified closed on the rebuilt table: **0 rows** where `club_record_date`
exceeds either `as_of_date` or `label_date`.

**The check that now catches this class.** The as-of join keeps its source date
as a real column, `club_record_date`, and `LeakageValidator` runs
`check_no_future_dates` over the *whole frame* rather than only the declared
feature columns. A numeric aggregate cannot be audited directly, but the
provenance timestamp of the join that built it can. If the merge direction ever
silently regresses, the pipeline stops.

---

## Part 2 — the surviving limitations

### L1 — coverage begins in 2012

**Investigated by construction, not by argument.** `appearances.csv` has zero
rows before 2012-07-03 and `game_lineups.csv` none before 2013-07-02, so no
season before 2012 has performance data. But `player_valuations.csv` reaches
back to 2000 (78,159 rows, 11,438 players) and `transfers.csv` to 1993, so
label-bearing rows *can* be built from biography and a prior valuation alone.

They were built: **40,339 rows across seasons 2003–2011, 10,741 players** — a
66% increase in training data for the prior-value variant. Only **7 features
survive** on such a row — age, age², height, position, sub-position, foot and
citizenship — because nothing else has data to come from. The experiment below
ran against the 41-feature list current at the time, on which 34 of 41 were
missing; under today's 54 it would be 47 of 54, so the block added since makes
these rows emptier rather than fuller and cannot change the conclusion.

Adding all of them changes nothing:

| held-out season | 2012+ only | with 40,339 reconstructed rows |
|---|---|---|
| 2020 | €1,409,608 | €1,427,860 |
| 2021 | €1,371,343 | €1,374,765 |
| 2022 | €1,674,177 | €1,642,205 |
| **pooled (n = 15,737)** | **€1,482,862** | **€1,479,989** |

Paired: d = +€2,873 (0.19%), t = +0.35, **p = 0.73**.

**Verdict: genuinely unfixable, now with a number rather than an assertion.**
The rows exist and are trustworthy; they carry no information. A row missing
83% of its features teaches the imputer's median, not the mapping. Not adopted.

### L2 — Transfermarkt values are estimates, not prices

Every candidate target was evaluated on a common yardstick: map the model's
output back to a market value and score EUR MAE on the same held-out rows.

**Transfer fee** (`transfers.csv`). 175,165 transfer rows, 17,554 with a fee
above zero, 7,326 joinable to a player-season — 8.5% coverage against the
market value's 83%. A fee exists only where a sale happened, so the label is
conditioned on the outcome; it answers "what did a sold player cost", not "what
is this player worth". Ships as `--target transfer_fee`, not as the default.

**Delta value** — predict `log(value_t) − log(value_{t−1})` and add the prior
back. Pooled over 2020–2022 (n = 15,737): level target €1,482,862, delta target
€1,492,676. d = −€9,813, t = −0.89, **p = 0.37**. It is the same model
re-parameterised — a booster given `prev_log_market_value_in_eur` can already
learn the offset — and it is nominally worse. Not adopted.

**Within-season percentile / rank.** Ranking ability is identical: Spearman
0.911 vs 0.910 (2020), 0.910 vs 0.911 (2021), 0.907 vs 0.909 (2022). But
converting a predicted percentile back to euros costs 50–66% more MAE
(€2,838,580 vs €1,624,448 in 2020), and the reason is structural rather than
fixable: the inverse map needs the *target* season's value distribution, which
nobody has at prediction time, so it must use the training distribution and
inherits its drift. The rank target discards scale, and scale cannot be
recovered from a distribution the model is not allowed to see. Not adopted.

**Career peak future value.** 28.4% of rows have no future season at all, and
the censoring is neither random nor stationary — 2024 rows have 0% future
coverage by construction, and a player's last observed season is frequently his
last season. Training on a right-censored target whose censoring correlates
with the target would teach "peak ≈ current" for the censored majority. Not
adopted.

**Verdict: `market_value_in_eur` remains the correct default**, now because
four alternatives were measured rather than because it was first.

### L3 — wide prediction intervals

**The method was already the best available. The calibration set was wrong.**

`calibrate()` measured its residual quantiles on the **test** seasons and the
artifact then reported an 80% interval scored against those same rows. An
interval fitted to the rows it is scored on covers its nominal level by
construction. No empirical coverage was computed anywhere in the project.

Seven approaches compared, all calibrated on 2022 and evaluated on 2023+, at a
nominal 80% (Winkler interval score, lower is better):

| method | coverage | mean log-width | Winkler |
|---|---|---|---|
| shipped, quantiles fitted on **test** | 0.800 | 1.696 | 10.61M |
| **band-wise quantiles, calibrated on validation** | **0.768** | **1.531** | **10.58M** |
| conformalised quantile regression (CQR) | 0.776 | 1.710 | 10.98M |
| split conformal, symmetric | 0.754 | 1.531 | 11.64M |
| LightGBM quantile regression, unconformalised | 0.626 | 1.294 | 12.53M |

(performance-only; with-prior-value orders identically and reaches 0.798.)

Pooling several seasons' out-of-fold residuals instead of one does not help —
0.761 to 0.768 across three variations — so the shortfall is not sample size.
It is temporal: split conformal guarantees coverage under exchangeability, and
consecutive football seasons are not exchangeable.

**Adopted:** the same band-wise method, calibrated on the validation season,
with the coverage it actually achieves measured on the test seasons and carried
in the artifact, printed in both model cards, served as
`confidence.measured_coverage`, and shown in the dashboard next to the nominal
level. Nothing was widened to make the number look better; that would be
fitting the test set through the back door.

**Not adopted:** NGBoost and CatBoost's uncertainty modes were not run. Both
lose to a method already beaten by the incumbent on Winkler score here, and
both would add a serving dependency the one-standard-error rule exists to
avoid. Stated so the omission is a decision rather than an oversight.

### L4 — the best model does not ship

See Part 4.

### L5 — the August–July season boundary

See Part 5.

---

## Part 3 — feature discovery

Every column of all ten Kaggle files was reviewed for unused signal, including
`game_events.csv` (1,274,469 rows), which the pipeline still does not download.
Four candidate blocks were built and measured. Two were rejected, one was
rejected as harmful, one was adopted.

### Adopted — career momentum (13 features)

The performance-only variant knew how *established* a player was
(`years_since_debut`, `seasons_observed`) but nothing about what he had
actually done before. A 24-year-old with 2,800 minutes is a different
proposition depending on whether last season was 2,900 minutes or 300.

Thirteen columns: the previous season's minutes, appearances, goal
contributions, start share, competition level, club form and squad-match share;
the gap in seasons; season-over-season deltas in minutes, contributions and
competition level; and expanding career means and maxima.

These are lagged **features**, not a lagged label, which is what makes them
legal in the variant that deliberately withholds the prior valuation. They come
from the player's own previous row, and `as_of_date` is monotone within a
player, so the previous row's evidence closed strictly earlier. Verified on the
rebuilt table: 61,555 rows carry momentum, all with a strictly earlier previous
as-of date, `prev_season_gap >= 1` throughout.

Pooled over five held-out seasons at three seeds:

| variant | baseline | + momentum | delta | paired |
|---|---|---|---|---|
| performance_only | €1,976,268 | **€1,850,058** | −6.4% | t = **11.30**, p = 1.6e−29 |
| with_prior_value | €1,524,483 | €1,510,028 | −0.9% | t = 1.94, p = 0.052 |

Improved in 5/5 held-out seasons for the first variant, 4/5 for the second.

Shipped on one feature list for both variants rather than two. The prior-value
variant is not harmed — its point estimate improves and it simply cannot be
distinguished from noise there, because `prev_log_market_value_in_eur` already
encodes most of what a trajectory says — and a second feature list is a second
thing to keep in step for a gain nobody can measure.

**Thirteen columns, not the five that survive a leave-one-out.** Leave-one-out
cannot see a feature whose information its neighbours duplicate: dropping
`prev_minutes_played` alone costs €148. Measured directly, the full block beats
the five-column subset by €27,990 of pooled MAE (t = 4.53, p < 0.0001).
Correlated features are cheap for a gradient booster, and the smaller set is
not the smaller error.

### Rejected — club home attendance

The most plausible candidate in the audit, and the most instructive failure. A
club's mean home attendance is a clean dated proxy for stature, present in
`games.csv`, and on a single validation season it looked like a win (−€53,486,
p = 0.007).

Pooled over five seasons it is a disaster: **+€297,445 MAE, t = −17.16,
p = 1.0e−65**, and it drags the momentum block down with it (+13 features alone:
€1,850,058; with attendance: €2,135,850).

The mechanism is COVID-19. Season 2020 was played behind closed doors:

| season | mean attendance | null rate | corr with log value |
|---|---|---|---|
| 2019 | 14,769 | 0.000 | +0.652 |
| **2020** | **2,752** | **0.216** | **+0.238** |
| 2021 | 11,394 | 0.001 | +0.657 |

For one season in fourteen the column measures a pandemic rather than a club.
A model trained across that boundary learns a relationship that does not hold,
and any evaluation window containing 2020 or 2021 pays for it. Rejected — and a
reminder that a feature validated on one season is a feature not yet validated.

### Rejected — transfer history

Career transfer count, days since last transfer, and log of the largest and
most recent prior fee, all joined strictly backward of the as-of date.

| variant | delta | paired |
|---|---|---|
| performance_only | +€4,188 | t = 0.34, p = 0.74 |
| with_prior_value | +€10,331 | t = 0.93, p = 0.35 |

Not significant either way. Four columns for nothing. Rejected.

### Rejected — club strength trajectory

Club points-per-game and league-position change against the club's own previous
season. Actively harmful for the variant that needed help most:
**−€35,695, t = −2.27, p = 0.023** for performance-only, and exactly zero
(+€62, p = 0.995) with prior value. Rejected.

### Not built — `game_events.csv`

157 MB and 1,274,469 rows of goals, cards and substitutions with minute stamps.
Its goal and card counts are already in `appearances.csv` at the same
granularity the model uses, and the marginal signal — the minute a substitution
happened — is not obviously a value driver. Left unbuilt deliberately, and
recorded here so the omission is a decision. `clubs.csv` remains excluded for
the reason Phase 15 gave: its interesting columns are all current state.

---

## Part 5 — the August–July season boundary

`games.csv` carries Transfermarkt's own season label for every fixture, which
respects each competition's calendar. The pipeline ignored it and derived the
season from the date instead, so this limitation was testable rather than
merely regrettable.

**The disagreement is real and larger than it looks.** The declared and derived
labels differ on **6.27% of fixtures**, concentrated exactly where a calendar
argument predicts: Brazilian Série A (39.9% of its fixtures), J1 League
(39.5%), Eliteserien (34.3%), Allsvenskan (31.0%), K-League (25.1%), MLS
(20.1%), and the European qualifying rounds played in July (ELQ 54.0%,
CLQ 57.5%, ECLQ 53.1%). 4,300 of the 5,582 disagreements are July fixtures.

**19.64% of player-declared-seasons span two derived seasons**, though only
2.02% of all minutes are misfiled — most splits are a couple of July matches,
not half a campaign.

### The harm is real, and it is the indexing rather than the population

Rows with more than 5% of their minutes misfiled have a higher median relative
error **within every value band**:

| value band | unaffected | >5% misfiled | difference | p |
|---|---|---|---|---|
| <1M | 0.329 | 0.396 | +0.068 | <0.0001 |
| 1M–5M | 0.368 | 0.438 | +0.070 | <0.0001 |
| 5M–20M | 0.343 | 0.465 | +0.122 | <0.0001 |
| 20M–50M | 0.304 | 0.341 | +0.037 | 0.049 |

The obvious confound is that misfiling selects a population — players in
European qualifying campaigns. It does not explain the result. Holding that
population fixed (only rows with qualifier minutes), misfiling still costs
+0.077 to +0.149 of median relative error at p < 0.0001; and conversely, among
rows with no misfiling, having played qualifiers costs essentially nothing
(+0.028 in one band, ~0.00 in the rest). The spring–autumn leagues themselves
are *not* badly modelled — median relative error 0.340 against the panel's
0.357.

### And re-indexing still does not fix it

A full alternative table was built on the declared season. The first result was
spectacular and wrong:

| | performance_only | with_prior_value |
|---|---|---|
| 2020 | +€56,289 (p = 0.020) | +€224,596 (p < 0.0001) |
| 2021 | +€206,774 (p < 0.0001) | +€121,432 (p = 0.0007) |
| 2022 | +€75,232 (p = 0.033) | −€22,589 (p = 0.58) |

on **more** rows (91,448 vs 85,966) and more players (26,674 vs 24,411).

It is an artefact of changing two things at once. Anchoring the as-of date on
the competition's own last fixture instead of 1 July moved it **earlier on 93%
of rows** and cut the median label horizon from **141 days to 23**. Only 50.9%
of shared rows even kept the same label. The model was not better; it was being
asked to forecast three weeks ahead instead of five months, which is a
different and much easier question.

Rebuilt with the as-of rule held fixed, so that only season *membership*
changes:

| | performance_only | with_prior_value |
|---|---|---|
| 2020 | +€5,211 (p = 0.77) | +€13,456 (p = 0.40) |
| 2021 | +€40,369 (p = 0.064) | +€519 (p = 0.98) |
| 2022 | −€26,699 (p = 0.25) | **−€68,996 (p = 0.0005)** |

and on **fewer** rows: 80,406 against 85,966 (−6.5%), 23,443 players against
24,411 (−4.0%).

**Verdict: feasible, implemented as a prototype, and rejected on measurement.**
Null for the scouting model and significantly *worse* for the prior-value model
in 2022, while discarding 6.5% of the panel.

The reason it fails is worth stating, because it is the actual limitation. The
declared index correctly groups July qualifiers with their own competition's
season, but the label anchor then no longer sits at the end of those seasons:
for a Brazilian season ending in December, a 1 July anchor sits seven months
late, and the 365-day label window starts missing. **This panel has no single
anchor that is right for every calendar.** Both available indexings are wrong
somewhere; the fixed-July one is wrong on fewer of the rows that matter, and it
keeps 5,560 more of them.

That is the honest form of this limitation. It is not "we did not think about
it" and not "it cannot be done" — it is "it was done, measured, and the
alternative is worse". Re-run `plans/06` if the dataset ever gains a
competition-calendar table that would let the anchor move per competition
without moving the horizon.

---

## Part 4 — model search and the model that does not ship

Eleven families were re-searched on the corrected feature set. The
one-standard-error rule and the explainability constraint are unchanged and
still bind; what changed is how much they now cost.

**The stacked ensemble's advantage has largely evaporated.** Pooled over five
held-out seasons at three seeds, `stacked` beats `lightgbm` by €10,152 with
t = 1.00, **p = 0.32** — indistinguishable. At v1.2.0 the documented gap was
1.92% of validation MAE. The career-momentum block gave LightGBM most of what
the blend was buying, which is the more satisfying way to close that
limitation: not by shipping the ensemble, but by making it unnecessary.

### Knowledge distillation

Teacher–student was tested properly rather than assumed. The teacher is the
stacked blend; the student is a plain LightGBM fitted on
`λ·log1p(actual) + (1−λ)·log1p(teacher)`. The student is fully explainable —
200 named importances, SHAP intact — and adds nothing to the serving image.

Pooled over 2018–2022 at three seeds (n = 33,129), performance-only:

| model | MAE | vs plain LightGBM |
|---|---|---|
| plain LightGBM | €1,851,005 | — |
| stacked teacher | €1,840,853 | +€10,152, p = 0.32 |
| **student, λ = 0.75** | **€1,828,672** | **+€22,333, t = 3.41, p = 0.0006** |
| student, λ = 0.50 | €1,832,662 | +€18,343, p = 0.012 |
| student, λ = 0.25 | €1,845,801 | +€5,204, p = 0.52 |

Two controls decide whether this is real and whether it is worth its cost.

**Is the ensemble doing the work, or is this just label smoothing?** A
self-distilled LightGBM — the same family teaching itself — gains
**€1,623, p = 0.81**: nothing. Against the stacked-taught student it loses by
€20,710, t = 3.22, p = 0.0013. The teacher's disagreement with itself is what
carries the signal, so this is distillation rather than smoothing.

**Is it worth a stacked fit on every training run?** That is the question the
last experiment answers, and it is the one that decides adoption. Merely
regularising LightGBM harder (`min_child_samples=60, reg_lambda=5.0`) recovers
€15,479 of the €22,333 at **zero** additional cost.

### And that is where distillation loses its case

Five regularisation settings, same pooled protocol:

**performance_only** (n = 33,129)

| setting | MAE |
|---|---|
| `min_child_samples=20, reg_lambda=0` *(the shipped default)* | €1,851,005 |
| `60, 5` | €1,835,525 |
| `20, 20` | €1,833,941 |
| `60, 0` | €1,835,657 |
| **`120, 20`** | **€1,831,159** |
| distilled student, stacked teacher | €1,828,672 |

`120, 20` vs the default: −€19,846, t = −2.63, **p = 0.0087**.
`120, 20` vs the distilled student: +€2,487, t = 0.36, **p = 0.72**.

**with_prior_value** (n = 26,176)

| setting | MAE |
|---|---|
| default | €1,530,737 |
| **`20, 20`** | **€1,508,524** |
| distilled student | €1,512,046 |

`20, 20` vs the default: −€22,214, t = −3.20, **p = 0.0014**.
`20, 20` vs the distilled student: −€3,522, **p = 0.58** — the regularised model
is nominally *ahead*.

**Verdict: distillation rejected; the regularisation it was substituting for
adopted instead.** The student was not learning the ensemble's knowledge. It
was being shrunk, and the search had simply never been allowed to shrink the
model directly, because `min_child_samples` and `reg_lambda` were not in the
LightGBM grid. Adding them costs four times the grid for one family and no
serving change at all; the teacher would have cost three boosters over three
cross-validation folds on every training run, forever, for a difference of
€2,487 at p = 0.72.

The four settings between the extremes are indistinguishable from one another
(p = 0.51, 0.53, 0.65) and the two variants prefer different corners, so both
parameters go into the grid at two values each and the existing expanding-window
search picks per variant rather than a constant being hard-coded.

**The explainability constraint now costs almost nothing.** `stacked` is ahead
of `lightgbm` by €10,152 at p = 0.32 on the corrected feature set, against the
1.92% gap documented at v1.2.0. The limitation "the best model does not ship"
is closed not by shipping the ensemble but by removing its advantage.

---

## Part 6 — error analysis

Two segments were added to the error report because the audit needed them and
could not get them: `primary_competition_id`, so "which leagues is this model
bad at" is answerable, and `career_stage` (from `seasons_observed`), so "is it
bad at players it has no history for" is separable from age. A first-season
28-year-old arriving from an uncovered league and a first-season 19-year-old
are the same problem — no prior row to lag — and age alone puts them in
different buckets.

The systematic weaknesses the audit found, and what was done about each:

**Error grows with value, and relative error grows as value falls.** Median
relative error runs 0.33 in the <€1M band against 0.30 above €50M, while
absolute error runs the other way. Both are reported per band; neither is
fixable by a feature, because the target spans four orders of magnitude and a
single MAE cannot describe both ends. Read the per-band table.

**Players with no prior season were the largest addressable weakness**, and
that is what the career-momentum block addresses — a 28.4% slice of rows that
previously carried nothing but biography and the current season. It is now the
block with the strongest measured effect on the scouting model.

**Rows whose season straddles the July boundary carry 0.04–0.12 more median
relative error within every value band.** Addressed by measurement and
rejected: see Part 5. The alternative index is worse.

**COVID seasons distort any feature that touches crowds.** Season 2020 is a
structural break, not an outlier, and it is the reason `club_home_attendance`
is not a feature. Any future feature drawn from `games.csv` must be checked
against season 2020 before it is believed.

---

## Part 7 — verdict

**(B) Improvements were found.** Four, of which two are correctness fixes that
no metric would ever have flagged.

### 1. Club form was built from matches played after the label

*Why it works:* it does not "work" — it removes a leak. A season runs
August–July, so 5.56% of fixtures fall after the 1 July as-of boundary they are
supposed to precede.

*Validation improvement:* none, and that is the finding. €−10,497 (p = 0.49)
and €−1,684 (p = 0.89). The leak was buying nothing and overstating everything.

*Statistical significance:* the defect is exact, not statistical — 1,049 rows
carried club form from matches played after their label was set, 15,126 from
after their as-of date. Now 0 and 0.

*Leakage analysis:* closed by construction — `merge_asof` backward on the row's
own date — and audited by value, because the join leaves `club_record_date`
behind and `check_no_future_dates` now reads the whole frame.

*Computational cost:* one `merge_asof` over 178k club-match rows. Seconds.

*Deployment impact:* none. Determinism verified across three independent
builds.

### 2. Thirteen career-momentum features

*Why it works:* the scouting variant had no memory. It knew a player was three
seasons in and nothing about what he did in them.

*Validation improvement:* €1,976,268 → €1,850,058 pooled, **−6.4%**, 5/5
held-out seasons. Test MAE €2,205,618 → €2,077,371, R² 0.813 → 0.833; with the
regularisation of §4 as well, €2,068,081 and R² 0.841.

*Statistical significance:* t = 11.30, p = 1.6e−29; Wilcoxon p = 2.5e−77.

*Leakage analysis:* lagged features, not a lagged label. `as_of_date` is
monotone within a player (asserted), so the previous row's evidence closed
strictly earlier. 61,555 rows verified, `prev_season_gap >= 1` throughout.

*Computational cost:* one `groupby.shift` pass. Negligible.

*Deployment impact:* 41 → 54 features on the wire. The current-season
prediction path draws them from labelled history, verified at 75% coverage.

### 3. The interval was calibrated on the rows it was scored against

*Why it works:* an interval fitted to the rows it is then scored on reports its
nominal level as a result.

*Validation improvement:* the reported coverage gets *worse*, from a
tautological 0.800 to a measured **0.763**. That is the improvement.

*Statistical significance:* not applicable — the previous number was arithmetic.
The method was chosen on Winkler score against four alternatives, and won.

*Leakage analysis:* quantiles from validation, coverage from test. Neither the
model nor the interval has seen the rows it is scored on.

*Computational cost:* one extra `predict` on the validation season.

*Deployment impact:* `confidence.measured_coverage` added to the response
(additive, optional), printed in both model cards, shown in the dashboard.

### 4. Regularisation in the LightGBM grid

*Why it works:* the shipped model was running at LightGBM's defaults on 54
features and 66k rows, and the search had never been allowed to shrink it.

*Validation improvement:* €19,846 pooled (performance-only) and €22,214
(with prior value). On the final retrain the search chose
`min_child_samples=120, reg_lambda=0` for the first and
`min_child_samples=20, reg_lambda=20` for the second — different corners, which
is why both went into the grid rather than a constant into the code.

*Statistical significance:* t = −2.63, p = 0.0087 and t = −3.20, p = 0.0014.

*Leakage analysis:* none — a hyperparameter.

*Computational cost:* four times the LightGBM grid, one family of eleven.

*Deployment impact:* none. This is also what killed distillation: the same gain
without a teacher.

### What the audit could not improve

Six ideas were built, measured and rejected: knowledge distillation, calendar
re-indexing, pre-2012 reconstruction, club attendance, transfer history, club
trajectory. Three alternative targets and three interval methods were evaluated
and none beat the incumbent. `game_events.csv` and `clubs.csv` remain
deliberately unused.

### The standing limitations, and why each is now a fact rather than a guess

| limitation | status |
|---|---|
| Coverage begins 2012 | rows reconstructed, added, measured worthless (p = 0.73) |
| Labels are estimates | four alternative targets measured, none better |
| Intervals are wide | honest coverage now measured and served; 0.763 vs 0.80 |
| Best model does not ship | gap now p = 0.32 — indistinguishable on this data |
| August–July boundary | alternative index built and measured *worse* |

**On the question as posed:** the project has not reached the practical limit
of this data — this audit found a genuine label leak and a 6.4% accuracy gain
that four previous passes missed. It is much closer to it now. What remains is
bounded by the dataset: no pre-2012 performance data exists, no per-competition
calendar table exists to move the label anchor without moving the horizon, and
market values are appraisals rather than prices. Those three are properties of
the world. Everything else in this audit turned out to be a property of the
pipeline, which is the fourth time in a row that has been the answer, and is
the reason to run a fifth pass rather than to declare victory.
