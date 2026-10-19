# Evaluation: what the metrics are evidence of

Written in Phase 20 in response to audit findings F5 (evaluation contamination)
and F12 (statistical correctness). It states what each reported number does and
does not demonstrate, and records the protocol under which an out-of-time number — one
whose labels did not exist when the predictions were made — can still be
obtained. No external replication has been attempted.

## 1. Status of every metric reported so far

**Every published accuracy and coverage figure is retrospective development
evidence, not an independent assessment.** The 2023+ "test" seasons were
inspected throughout development, and at least three design decisions were made
with test results in view. Retraining cannot undo that: a model built by a
process that has seen the test set cannot be re-scored on it as though it had
not.

The figures remain worth reporting. They come from seasons later than every
season the models were *fitted* on, under a strict as-of discipline that the
leakage stage enforces row by row. That makes them a fair description of how the
current models behave on 2023–2024. What they are not is evidence about a season
nobody had looked at.

### Inspection history of the 2023+ seasons

| Phase | What was done with the test seasons | Effect on design |
|---|---|---|
| 0 — discovery (2026-08-28) | Random vs group vs temporal split compared on 2023+ | Chose the temporal split as the headline protocol |
| 15 — `plans/05` | `label_horizon_days` as a feature: test R² 0.760 with it, 0.762 without | **Decision taken on test:** the feature was dropped |
| 16 — `plans/06` | Seven interval methods calibrated on 2022 and **scored on 2023+** | **Decision taken on test:** the band-wise method was adopted |
| 16 — `plans/06` | Momentum block, regularisation: chosen on 2018–2022 folds, test MAE reported alongside | Reported, not decisive |
| 17 — `plans/07` | Test MAE reproduced; calibration-set reuse examined on test coverage | Diagnostic |
| 18 — `plans/08` | Tuning objective fixed; test MAE moved the wrong way and the fix was kept | Inspected, explicitly not decisive |
| 19 — `plans/09` | Test metrics read after each retrain; Random Forest vs LightGBM known before the serving policy | The policy was set on serving cost, but by someone who knew the forest's test score |
| 20 — this phase | 75.1% interval coverage diagnosed by season and band on test | **Motivated** the F9 quantile widening; its rule was fixed on pre-test seasons only |

The protocol in §2 was written and committed before the models in §3 were
trained, and nothing in it was chosen from the numbers in §3. That reduces, but
does not remove, the contamination above, which is part of how the protocol came
to be.

## 2. The frozen modelling protocol

Frozen at commit `b8d7d0f` (code that built the features and trained the shipped
models). Any change to an item below is a new protocol version, and a prospective
freeze (§4) made under one version says nothing about another.

| Element | Definition | Where |
|---|---|---|
| Unit | One player-season; season = August to July | `assign_season` |
| As-of date | `max(1 July season boundary, last appearance)` | `build_training_table` |
| Label | First valuation on or after the as-of date, within 365 days; fitted as `log1p(EUR)` | `attach_label` |
| Features | 54 (performance-only) or 56 (with prior value: latest valuation strictly before the as-of date, and its age) | `FEATURE_COLUMNS`, `select_variant` |
| Point-in-time rule | Every windowed feature carries the date of its newest input; events must be on or before the as-of date, labels strictly before, and an undated value is an error | `FEATURE_PROVENANCE`, `src/validation/leakage.py` |
| Split | Train ≤ 2021, validation 2022, test ≥ 2023 | `configs/config.yaml` |
| Hyperparameters | Per family, grid searched on expanding-window season folds inside the training seasons, recency-weighted | `src/models/tuning.py`, `src/models/weighting.py` |
| Family selection | Eligible = feature importances, SHAP explains every timed validation row, predict+explain ≤ 100 ms p50; one standard error of the best eligible validation MAE; smallest deployed footprint | `SELECTION_RULE` in `src/pipelines/tune.py` |
| Final fit | Winner refitted on the training seasons only | `_finalise` |
| Intervals | Band-wise residual quantiles on the validation season, at the smallest level (0.80–0.95) whose mean coverage reaches 80% when replayed on origins 2016–2020, one and two seasons ahead | `select_quantile_level` |
| Evaluation | Test seasons scored once per training run; reported as retrospective (§1) | `train_variant` |
| Provenance | Raw files, training table, configuration, code commit and environment hashed into every artifact | `src/models/provenance.py` |

## 3. Retrospective results, with honest uncertainty

Computed by `scripts/evaluate_retrospective.py` (committed before it was run),
training table SHA-256 `5d65c52a68dff3bd…`, backtest predictions
SHA-256 `483acf62511b7b20…`. Intervals are 95% percentile
intervals from 2,000 bootstrap draws **over players**; within a single season each
player appears once, so per-season intervals are effectively row-level.

### 3.1 The shipped models on the test seasons

| Variant | Test MAE | 95% CI (players resampled) | R² | Rows |
|---|---|---|---|---|
| performance_only | €2,057,285 | €1,958,861 – €2,153,692 | 0.841 | 13,486 |
| with_prior_value | €908,625 | €861,901 – €957,500 | 0.963 | 13,315 |

These are the numbers the README quotes. Fitted on seasons up to 2021,
selected on 2022, scored on 2023–2024; retrospective, for the reasons in §1.

### 3.2 Rolling-origin backtest of the frozen configuration

Each season *s* is predicted by the shipped family and hyperparameters refitted on
every season before *s*. It describes how the configuration behaves across
seasons; it is not a test of the selection procedure, which was not re-run per
origin, and the configuration itself was designed with these seasons in view.

**performance_only**

| Season | MAE | 95% CI | R² | Rows |
|---|---|---|---|---|
| 2016 | €1,394,338 | €1,302,975 – €1,487,445 | 0.781 | 6,530 |
| 2017 | €1,907,025 | €1,779,925 – €2,036,624 | 0.750 | 6,630 |
| 2018 | €2,162,655 | €2,024,308 – €2,294,468 | 0.760 | 6,648 |
| 2019 | €1,747,551 | €1,649,101 – €1,852,845 | 0.823 | 6,467 |
| 2020 | €1,496,988 | €1,416,845 – €1,583,637 | 0.850 | 7,216 |
| 2021 | €1,752,486 | €1,660,499 – €1,846,339 | 0.848 | 6,225 |
| 2022 | €1,892,889 | €1,794,430 – €1,997,135 | 0.837 | 6,573 |
| 2023 (test) | €2,101,844 | €1,987,380 – €2,213,886 | 0.861 | 5,726 |
| 2024 (test) | €1,834,029 | €1,740,187 – €1,934,064 | 0.845 | 7,760 |

**with_prior_value**

| Season | MAE | 95% CI | R² | Rows |
|---|---|---|---|---|
| 2016 | €756,093 | €699,408 – €813,894 | 0.911 | 6,427 |
| 2017 | €847,232 | €788,368 – €917,738 | 0.938 | 6,535 |
| 2018 | €964,648 | €902,893 – €1,026,898 | 0.952 | 6,573 |
| 2019 | €754,383 | €710,209 – €800,045 | 0.968 | 6,332 |
| 2020 | €723,865 | €685,032 – €767,189 | 0.968 | 7,129 |
| 2021 | €805,702 | €759,831 – €855,643 | 0.963 | 6,200 |
| 2022 | €900,232 | €850,068 – €954,189 | 0.960 | 6,493 |
| 2023 (test) | €957,022 | €904,212 – €1,014,851 | 0.967 | 5,666 |
| 2024 (test) | €866,052 | €818,484 – €918,306 | 0.963 | 7,649 |

**Season-to-season variation dominates.** Performance-only MAE ranges from
€1.39M (2016) to €2.16M (2018), while each season's own interval is about
±€100k. Any single season — or the two test seasons — is a small sample of
seasons, and the per-season series is reported rather than one pooled number
because overlapping training windows make the seasons dependent.

### 3.3 Design effects: how much row-level tests understated uncertainty

| Pooled over | Variant | Players | Rows | Design effect |
|---|---|---|---|---|
| 2016–2022 | performance-only | 16,065 | 46,289 | 3.79 |
| 2016–2022 | with prior value | 15,814 | 45,689 | 3.11 |
| 2023–2024 | performance-only | 8,612 | 13,486 | 1.51 |
| 2023–2024 | with prior value | 8,488 | 13,315 | 1.53 |

The design effect is the clustered variance of the mean absolute error over the
row-level variance. Pooled over seven seasons it is about 3–4, so a row-level
standard error is roughly half what it should be, and a row-level *t* of 3 is
nearer 1.6 once players are respected. This is measured on MAE levels; for a
paired *difference* between two models the correlation within a player may be
lower, so treat it as indicative for §5's annotations rather than exact.

### 3.4 The two variants, paired and unpaired

Unpaired, each on its own rows (§3.1): performance-only on 13,486 rows,
prior-value on 13,315. Those two MAEs are not a comparison: the prior-value
variant drops players never valued before their as-of date.

Paired, on the 13,315 test rows both variants score (8,488 players):
the prior value lowers MAE by **€1,171,403** (95% CI €1,098,515 –
€1,242,641), 56% of the performance-only error on those rows.
Both variants are LightGBM with the same feature list apart from the two prior
columns, so this is what the prior valuation is worth, not a better algorithm.

**A newer prior, not a better model (carried-over risk).** The prior-value
headline fell from €1.64M to €0.91M in Phase 19 (an unpaired comparison: 10,169
rows then, 13,315 now) because the prior was redefined
as the latest valuation before the as-of date (median 33 days old, against 242
under the old definition), not because anything about the estimator improved;
`plans/09` holds the controlled comparison. Error by the prior's age is in the
study output but confounded — rarely revalued players are cheap ones, so older
priors show *smaller* euro errors — and is not evidence either way.

## 4. A prospective out-of-time holdout, not yet scored

**An out-of-time holdout exists and has not been looked at.** "Independent" here
means independent in time: the labels did not exist in the data when the
predictions were made. It does not mean an external party — the same author
built the models, froze the predictions and will score them, so a scored
freeze is prospective validation, not independent replication. The local snapshot's
newest valuation is dated 2026-06-12. The current-season rows (season 2025, 8,709
rows) have an as-of date of 2026-07-01, so every one of their labels was published
after the data anybody working on this project has seen.

`scripts/prospective.py freeze` predicts those rows with the shipped models and
writes the predictions with a SHA-256 of them, of each model, and of each input
table (`reports/prospective/<date>/`). The freeze of 2026-10-18 — 8,709 performance-only and 8,131 prior-value
predictions, SHA-256 `cd71dbd3…348c1` — is recorded in
[`prospective/2026-10-18-freeze.json`](prospective/2026-10-18-freeze.json) and was
committed (`250b85c`) **before** any newer data was downloaded. That commit is
local: the branch has not been pushed, and a git commit date is whatever its
author's clock said, so it is not a public record or a third-party timestamp.
It becomes evidence a reader can check once it is published to a public host
(or timestamped externally) *before* the outcomes are downloaded; until then
the freeze date is self-reported. The predictions file is not in the repository;
it must be kept to be scored, and `score` refuses any file with another hash.

`scripts/prospective.py score <freeze> <newer player_valuations.csv>` refuses a
predictions file whose hash has changed, rebuilds each label with the training
rule, and reads only valuations published after the cutoff. Rows whose 365-day
window is still open and unlabelled are reported as **pending**: until the last
window closes (2027-07-01), the scored rows are early revaluations, which are not a
random subset, and the output says so.

**Practical:** the newer valuations come from the same weekly Kaggle refresh,
downloaded into a separate directory so the training snapshot is untouched.
**Legal:** the same uncertainty as the training data (`docs/DATASET_CARD.md`,
"Data rights") applies; obtaining the outcomes does not resolve it.

**Not done in this phase.** No newer data has been downloaded and nothing has been
scored. F5 is therefore **not** scientifically resolved: the independent
assessment is prepared, not completed.

## 5. Statistical review of published comparisons (F12)

**How every earlier p-value was computed.** Phases 16–18 compared configurations on
expanding-window seasons 2018–2022, pooled the errors across seasons, and ran a
paired *t*-test on absolute errors (with a Wilcoxon test where noted). Both treat
each row as independent. Pooled over five seasons, a player can contribute up to
five rows whose errors move together, so those standard errors are too small and
the p-values too optimistic. The size of that understatement is measured in §3 as a
design effect.

How to read the published figures in the light of that:

- **Very large effects survive any plausible design effect.** The momentum block
  on the performance-only variant (t = 11.3) and the rejection of club home
  attendance (t = −17.2 against it) are not in doubt.
- **Marginal p-values should not be read as significant.** Anything near
  p = 0.05 to 0.001 under a pooled row-level test — club strength trajectory
  (p = 0.023), the regularisation gains (p = 0.0087, 0.0014), the distilled
  student (p = 0.0006), the momentum block on the prior-value variant
  (p = 0.052) — is
  unconfirmed under player clustering. Single-season comparisons (attendance's
  p = 0.007 on 2022 alone) repeat no player, so clustering does not inflate them,
  but one season is one draw of everything season-wide.
- **A non-significant result is absence of evidence, not equivalence.** "Cost
  nothing measurable", "indistinguishable" and "the constraint is free" mean the
  test did not detect a difference at its row-level resolution. They do not show
  the difference is zero, and the documents now say so where they made the claim.
- **Different populations are compared unpaired.** The two variants are scored on
  different rows (the prior-value variant drops players never valued before), and
  Phase 19's before/after figures used different row sets. Those comparisons are
  labelled unpaired; the paired comparison on shared rows is in §3.
- **Seasons are not independent replicates.** Expanding-window backtests share
  training data and players, so per-season results are reported side by side
  rather than combined into one test statistic.

The experiments behind the Phase 16–18 figures were run from scratch scripts and
cannot be re-run on the current code without re-creating each historical state,
so their p-values are annotated rather than recomputed. Everything from Phase 20
onward is computed by committed code: `src/evaluation/inference.py` and
`scripts/evaluate_retrospective.py`.
