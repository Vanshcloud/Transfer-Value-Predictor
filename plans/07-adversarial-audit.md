# Phase 17 — adversarial audit

Phase 16 tried to improve the project. This one tries to break it.

Independent-reviewer posture: every claim treated as false until the repository
proves it, every verification re-derived from the raw files rather than from the
pipeline's own checker, and the previous phase's changes attacked as hard as
everything else. Four of the eight findings below are in code Phase 16 wrote.

Nothing here was found by reading. Every finding is a measurement.

## Findings

| # | Severity | Area | Status |
|---|---|---|---|
| 1 | **High** | `similar_players` empty for 32.8% of players | fixed |
| 2 | **High** | `dropna` over all features hid 65.3% of rows | fixed |
| 3 | Medium | Leakage validator never saw 28.4% of rows, or the serving table | fixed |
| 4 | Medium | `min_age`/`max_age` declared, validated, never used | fixed |
| 5 | Low | Five stale numeric claims in docstrings | fixed |
| 6 | Low | "34 of 54 features" where the measurement was 34 of 41 | fixed |
| 7 | Info | Tuning fits unweighted, final fit weighted | measured |
| 8 | Info | 2 validation rows with a 1-day label overlap; 0 test rows | no action |

---

### 1 — `similar_players` returned nothing for every active player

**Severity: High.** Silent: the endpoint returned `200` with `[]`, and the
dashboard rendered "No comparable seasons found."

**File:** `src/services/prediction.py`, `similar_players`.

**Evidence.** The method anchored on `rows.iloc[-1]["season"]` — the player's
most recent row. Since the current-season table arrived in v1.2.0, that row is
the season being played, which carries no valuation. The neighbour pool is
labelled-only, so:

```
players whose LAST row is the unlabelled current season: 8,709 of 26,592 (32.8%)
labelled pool for season 2025: 0 rows  ->  similar_players returns []
labelled pool for season 2024: 7,760 rows
```

Measured directly: 0 of 200 sampled current-season players got comparables.

**Impact.** The comparables feature was dead for exactly the population anyone
looks up — everybody currently playing. A user searching an active player saw
an empty panel indistinguishable from "this player has no comparables".

**Fix.** Anchor on the player's latest *labelled* season. A comparison needs a
market value to display on both sides, so the anchor must be a season that has
one. After: 144 of the same 200 return comparables; the remaining 56 are
players whose only season is the one being played, for whom `[]` is the honest
answer and is now asserted as such.

### 2 — `dropna` over every feature, in a pipeline that imputes

**Severity: High.** Also silent, also an empty list.

**Files:** `src/services/prediction.py`, `_rows_for_variant` and
`prediction_history`.

**Evidence.** Both required all 54 features to be non-null before a row could
be shown, while the fitted pipeline carries a `SimpleImputer` that was trained
to handle exactly those nulls.

```
labelled rows dropped:                    56,148 of 85,966  (65.3%)
of which a player's FIRST season:         24,411
players left with no usable row at all:   14,402 of 24,411  (59.0%)
```

Not a new bug — the pre-audit 41-feature list dropped 56.5% and left 11,840
players empty — but Phase 16's momentum block made it worse, and
`second_half_goal_share` alone is null on 51.3% of rows because a player who
never scored twice in a season has no second-half goal share to have.

**Impact.** 59% of players had an empty career chart. Every player's first
season was missing from his own history. Henrikh Mkhitaryan's chart began in
2014 for a career on record from 2011 — and the dropped 2011 row predicts
perfectly well (EUR 3,721,074 against an actual EUR 12,000,000) because the
imputer does its job.

**Fix.** A display gate rather than a completeness gate: the target, plus the
prior value for the variant defined by it. Nothing else. After: 85,966 rows and
all 24,411 players for the scouting model; Mkhitaryan gets all 14 seasons.

### 3 — the leakage validator never saw 28.4% of the rows

**Severity: Medium.** A coverage gap, not an active defect: the unchecked rows
pass when checked.

**File:** `src/pipelines/features.py`, `build_features`.

**Evidence.** The stage validated `select_variant(include_prior_value=True)`
on the reasoning that it "carries every column the performance-only variant
does". True of the *columns*, false of the *rows* — that frame is 61,522 of
85,966, so 24,444 rows were never checked for duplicate entities, feature/label
ordering, or future-dated values. The current-season table, which is what
`/api/v1/predict` answers from, had no validation at all.

Running the validator on the excluded rows: **0 errors, 0 warnings**. On the
serving table: **0 errors, 0 warnings**. Both pass. Nothing was making them.

**Fix.** Validate the full table, the prior-value frame, and the serving table.
All three now run in the pipeline and all three are clean.

### 4 — two config keys that were parsed, cross-validated, and read by nothing

**Severity: Medium.**

**Files:** `configs/config.yaml` (`data.min_age`, `data.max_age`),
`src/utils/config.py` (including a `max_age > min_age` validator).

**Evidence.** `grep` finds no reader outside `config.py` itself. The
consequence reached the API:

```
POST /api/v1/predict {"features": {"age": 1e12}}  ->  200 OK, a confident number
```

which is the failure mode `NON_NEGATIVE_FEATURES` exists to prevent, one field
over — the project's own words: "a negative goal count is not an unusual
player; it is a caller bug, and answering it with a confident number is the
failure mode this project spends most of its effort avoiding."

**Fix.** `PLAUSIBLE_RANGES` in `src/feature_engineering/build.py`, beside
`NON_NEGATIVE_FEATURES` for the reason that module already gives — it is a
property of what the feature means, so it must hold for a batch job too.
Verified safe before enforcing: **0 of 85,966 training rows and 0 of 8,709
current-season rows** fall outside age [15, 45] or height [140, 220].
`tests/unit/test_config.py` now asserts the config and the enforced bound
cannot drift apart.

### 5 — five numeric claims in docstrings that no longer hold

**Severity: Low.** Explanatory prose; no behaviour depends on them. Included
because the project's stated standard is that every figure is reproducible, and
`tests/unit/test_documented_numbers.py` exists precisely because "prose does
not recompute".

| file | claimed | measured |
|---|---|---|
| `src/models/baseline.py` | target raw skew 8.70 | **6.16** |
| `src/evaluation/metrics.py` | target raw skew 8.70 | **6.16** |
| `src/feature_engineering/build.py` | prior-value skew 4.53 → 0.10 | **5.53 → 0.31** |
| `src/models/splits.py` | R² 0.465 / 0.455 / 0.412, "60% worse" | **0.805 / 0.795 / 0.770, 45% worse** |
| `src/pipelines/tune.py` | explainability costs 1.92% / 0.33% | **1.09% / −0.32%** |

Every conclusion survives — log1p still removes most of the skew, temporal is
still much worse than grouped, the ordering never changed. The last one shrank
rather than flipped: on the prior-value variant the top three families now sit
inside EUR 5,600 of one another, a tenth of the validation standard error, so
the constraint is settled by the deployment-cost tiebreak rather than by
accuracy. (Phase 18 retrained under a corrected search objective; the figure
quoted in that table is the one from this phase's run.)

Claims that survived: appearances start 2012-07-03 (exact), `contract_expiration_date`
37% null (37.03%), height 1.2% null (1.08%), foot 1.7% null (1.57%), log1p
target skew 0.43 (0.37), label coverage 90.8% (90.8%).

### 6 — "34 of 54 features are structurally missing"

**Severity: Low.** `README.md` and `docs/DATASET_CARD.md` attached a
41-feature measurement to a 54-feature list. The true figure is that **7 of 54
features exist** on a reconstructed pre-2012 row; under the old list 34 of 41
were missing. Corrected in all three places, with the provenance of the
measurement stated so it cannot drift again.

### 7 — the search and the final fit optimise different objectives

**Severity: Informational.** Real, and not leakage.

**Files:** `src/models/tuning.py:_score_candidate` fits
`pipeline.fit(train[features], train[target_column])`;
`src/pipelines/tune.py:_tune_and_validate` and `_fit_winner` fit with
`**fit_params(train["season"])` — recency weighting.

So hyperparameters are chosen under an unweighted objective and the winner is
then fitted weighted. Reported rather than changed: see the measurement below.

### 8 — a one-day label overlap on two validation rows

**Severity: Informational, no action.** Under a strict
"simulate-deployment-at-time-T" reading, a model fitted on training labels
published up to 2023-07-04 is scored on two validation rows labelled 2023-07-03.

```
validation: 2 of 6,573 rows (0.03%), overlap 1 day
test:       0 of 13,486 rows (0.00%)
```

One training label of 65,907 postdates the earliest validation as-of date; none
postdates the earliest test as-of date. The reported headline metric is
computed entirely on the test seasons and is unaffected.

---

## What was attacked and survived

Recorded so the absence of a finding is a result rather than a gap in effort.

**Leakage.** Every `merge_asof`, expanding window, cumulative statistic,
`groupby().transform()` and target encoding was re-derived independently.
`competition_value_level` reproduces a brute-force strictly-prior recomputation
exactly (341 comparisons, 0 mismatches, max diff 0.00e+00). Prior values trace
to a real raw valuation on 61,522 of 61,522 rows. `club_record_date` never
exceeds `as_of_date` or `label_date` on any of 85,966 training or 8,709 serving
rows. No duplicate `(player_id, season)`.

**Preprocessing.** Proven numerically, not by inspection: the fitted imputer's
medians match the *training* rows exactly (max diff 0) and differ from the full
table by 28; likewise the scaler's means. Imputer, scaler and one-hot encoder
all live inside the `Pipeline`, so `fit` cannot see validation or test.

**Chronology, per partition.** `feature_time <= label_date`,
`as_of <= label_date`, `as_of >= last_appearance`, `club_record <= as_of` and
`prev_value_age_days > 0` all hold on train, validation, test and the
prediction table independently. Label ranges are ordered: train ends
2023-07-04, validation 2023-07-03→2024-06-28, test 2024-07-01→2026-06-09.

**Reproducibility.** The feature build is bit-identical across processes with
`PYTHONHASHSEED=random`. Both shipped models refit bit-identically from
scratch, reproduce their recorded test MAE to the cent, and predict identically
twice. `RandomForest`/`ExtraTrees` remain non-bit-reproducible under `n_jobs=-1`
and are documented as such in the registry; neither ships. CI and Docker do not
train, so neither affects model reproducibility — Docker's `python:3.13-slim`
base is a minor tag rather than a digest, which affects the serving image only.

**Evaluation.** No reported metric touches training rows: every `evaluate` call
in `src/pipelines/` scores `split.validation` or `split.test`, and SHAP is
computed on test rows. The interval's coverage is measured on the test seasons,
which neither the model nor the interval has seen.

**Calibration-set reuse.** The validation season selects the family *and*
supplies the interval's quantiles. Examined: coverage measured against
quantiles from a season never used for selection is 0.740 / 0.777, against
0.769 / 0.793 for the shipped ones — the reuse does not inflate the figure, and
could not, because coverage is measured on the untouched test seasons either
way.

**API.** Twenty adversarial requests: unknown player, negative and non-integer
ids, empty body, unknown feature key, negative counts, NaN and infinity, an
unknown variant, a season not on record, a 5,000-character category, both id
and features, an injection-shaped search, an oversized limit. Every one returns
a typed error with the correct status. The only input that got through was
`age: 1e12` — finding 4.

**Code quality.** No TODO, FIXME, HACK or commented-out code anywhere in `src`,
`api`, `tests`, `scripts` or `frontend/src`. No unused `__all__` export. No
unused config key after finding 4. `vulture` at 80% confidence: clean.

## Verdict — C: production-ready with documented limitations

The methodology is sound and the implementation now matches it. Leakage,
chronology and reproducibility are clean under adversarial testing, and every
documented figure reproduces.

It is **not** D, and the reason is not the modelling:

1. **A fifth audit still found two high-severity bugs**, both silent, both
   returning `[]`. 724 tests did not catch them because no test asserted that a
   *populated* result comes back — only that the shape is right. That is a
   structural blind spot in the suite, narrowed here but not closed.
2. **The headline metric rests on one split.** Selection uses a single
   validation season and the test set is scored once. That is the right
   discipline for deployment and thin for a paper, where the repeated-season
   protocol used throughout Phases 16–17 for *comparisons* should be the
   headline for the *result* too.
3. **No external baseline.** Nothing here is compared against a published
   market-value model, so "good" is established against this project's own
   earlier selves.

The remaining data limitations are intrinsic and measured, not implementation
defects: no pre-2012 performance data exists, no per-competition calendar table
exists that would let the label anchor move without moving the forecast
horizon, and Transfermarkt market values are appraisals rather than prices.
