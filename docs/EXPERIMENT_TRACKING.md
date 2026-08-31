# Experiment tracking

Phase 7 trains nine model families across two variants with a hyperparameter
search behind each one. That is where "which run was best?" stops being
answerable from memory. This document fixes what a run *is*, so any recorded
number can be traced back to the exact thing that produced it.

Nothing here is aspirational. Every field listed is written into the saved
artifact by `src/models/artifact.py`, and a test asserts a reloaded artifact
reproduces its recorded metrics exactly.

**No MLflow, no Weights & Biases.** One run per model family per variant, a few
dozen fits in total, all on one machine. A tracking server would be a service
to run, a schema to migrate and a second source of truth to reconcile — for
something a joblib file and a JSON sidecar already answer. Revisit when runs
outlive a single machine or several people launch them concurrently.

---

## 1. Dataset version

The upstream Kaggle dataset `davidcariboo/player-scores` refreshes weekly, so
"the data" is not a fixed thing and a metric without a dataset stamp is not
reproducible.

| Field | Meaning |
|---|---|
| `dataset` | Kaggle slug, from `configs/config.yaml` |
| `source_rows` | Row count of each ingested raw table |
| `built_at` | UTC timestamp of the feature build |

A metric compared against another metric built from a different `source_rows`
is a comparison across two datasets, and must be labelled as one.

## 2. Feature set version

| Field | Meaning |
|---|---|
| `variant` | `performance_only` or `with_prior_value` |
| `feature_columns` | The exact ordered list handed to the model |
| `n_rows` | Rows in the variant after filtering |
| `target_column` | `market_value_in_eur`, always |

The variant is not a hyperparameter. They are two products answering different
questions (00-discovery.md §2.3) and their metrics are never pooled or ranked
against each other — `performance_only` is the scouting model, and it is
*supposed* to score worse.

Changing `feature_columns` invalidates every earlier run. Adding a feature is a
new feature-set version, not a tweak.

Feature-set versions so far: 19 columns (v1.0), 41 (v1.2, the seven unused
Kaggle files), 54 (v1.3, the career-momentum block).

## 3. Split strategy

**Temporal is the only split that reports.** Train ≤2021, validate 2022, test
2023+, from the `split` block of `configs/config.yaml`.

| Stage | Rows used | Touched |
|---|---|---|
| Hyperparameter search | train seasons, expanding-window folds | many times |
| Model family selection | validation season (2022) | once per family |
| Final reported number | test seasons (2023+) | **once, at the end** |

Group and random splits are diagnostics. They exist so the gap to the temporal
number can be quoted rather than asserted, and they are never the headline: a
random split on a time series answers how well the model interpolates among
seasons it has already seen, which is not a question anyone deploying it asks.

Cross-validation inside the training seasons is **expanding-window**, never
K-fold. K-fold on a time series trains on 2023 to predict 2019.

## 4. Metric to optimise

**Selection metric: MAE in EUR on the validation season.**

MAE rather than RMSE because the business question is "how far off is a typical
valuation", and RMSE lets a handful of €100M outliers choose the model. EUR
rather than log space because log compresses exactly the expensive mistakes at
the top of the market — a model can look better in log space while being worse
in every euro that matters.

R², RMSE and MAPE are recorded alongside, and reported. They do not select.

### A standing caution

Do not chase R² at the cost of correctness. A temporal R² of 0.77 is worth more
than a random-split 0.82, because only the first one describes what happens on
a season the model has not seen. If a change improves the headline number,
check what it did to the leakage report before believing it.

### A second standing caution: check what question the change is answering

The final audit prototyped indexing seasons by each competition's own calendar
instead of a fixed August–July boundary. It looked like a 10% win — up to
€206,774 of held-out MAE, p < 0.0001, on more rows and more players.

It was an artefact. Anchoring the as-of date on the competition's last fixture
instead of 1 July moved it earlier on 93% of rows and cut the median label
horizon from **141 days to 23**. The model was not better; it was being asked
to forecast three weeks ahead instead of five months. Re-run with the as-of
rule held fixed so that only season *membership* changed, the same experiment
returns p = 0.77 and p = 0.06 and costs 5,560 rows.

So: when a change moves a metric a long way, find out whether it also moved the
question. Anything that touches `as_of_date`, the label window, or which rows
survive the join changes the difficulty of the problem, and a difficulty change
shows up in a metric exactly like a modelling improvement. The cheap diagnostic
is `label_horizon_days` — compare its distribution before and after.

### A third: validate on more than one season

`club_home_attendance` improved the 2022 validation season by €53,486
(p = 0.007) and cost €297,445 (p = 1e−65) pooled across 2018–2022, because
season 2020 was played behind closed doors and the column measures a pandemic
for one year in fourteen. One held-out season is an anecdote.

## 5. Random seed

`RANDOM_SEED = 42`, defined once in `src/models/splits.py` and passed to every
splitter and every estimator that accepts one. Two runs must agree **exactly**,
not approximately; `tests/integration/test_baseline_metrics.py` asserts it.

The sibling project shipped fourteen days of unreproducible metrics before
anyone noticed a missing seed. That is the failure this rule exists to prevent.

**One measured exception.** `random_forest` runs with `n_jobs=-1`, and the
order in which the average over trees is reduced is not fixed, so two
predictions from the *same* fitted forest differ by roughly 1e-15 relative. The
trees are seeded and identical; only the summation order moves. Making it
bit-exact means `n_jobs=1`, measured at 101s per fit against 23s — 27 minutes
for that one family across the search. So its reproducibility is asserted at
`rtol=1e-9`, which is still nine orders of magnitude tighter than anything a
missing seed could survive. Every other family is asserted bit-identical.

## 6. Hyperparameter search strategy

Exhaustive grid over a deliberately small grid per family — two to four
configurations each, scored by mean MAE across the expanding-window folds.

Small grids on purpose. A large grid over nine families is many hundreds of
fits for a gain that the temporal split will mostly wash out anyway, and every
extra configuration is another chance to overfit the validation season. The
baseline the zoo has to beat is already recorded (Phase 6): if a tuned model
cannot beat an untuned `HistGradientBoostingRegressor`, more search is not the
missing ingredient.

Search happens on training seasons only. The validation season is scored once
per family, after its hyperparameters are frozen.

## 7. Best-model tracking

One artifact per variant, written to `models/`, containing:

| Field | Why it is there |
|---|---|
| `pipeline` | The fitted preprocessing **and** estimator, as one object. Preprocessing fitted separately is the classic serving skew. |
| `model_name` | Which family won |
| `params` | The winning hyperparameters |
| `metrics` | Validation and test metrics, in EUR |
| `feature_importance` | Named, post-preprocessing, so it is readable |
| `feature_columns` | Input contract for the API in Phase 9 |
| `dataset` / `split` / `seed` | Everything in §1–§5 |
| `leaderboard` | Every family's validation score, so a later "which was best?" is answerable |
| `artifact_version` | Schema version of this file |

The artifact is self-describing: loading it requires no reference to the config
that produced it. `tests/integration/test_model_artifacts.py` asserts a
reloaded artifact reproduces its recorded metrics exactly, which is the only
thing that makes any of the above trustworthy.

## 8. What is deliberately not tracked

- **Per-fit wall-clock time.** Interesting, not decision-changing here.
- **Full CV fold predictions.** Regenerable from the seed.
- **Intermediate search results.** The leaderboard keeps the per-family score;
  every configuration tried is noise once the winner is known.
