# Transfer Value Predictor

Predict the market value (EUR) of professional footballers from performance,
biographical and contextual data — and explain every prediction.

> **Status: Phase 7 of 13.** Ingestion, validation, the training table, the
> baselines and the tuned model zoo are built. The API and dashboard are not. See
> [`plans/IMPLEMENTATION_PLAN.md`](plans/IMPLEMENTATION_PLAN.md).

## Why this repository looks the way it does

The architecture was decided *after* examining the data, not before. Three
findings shaped it, all recorded with evidence in
[`plans/00-discovery.md`](plans/00-discovery.md):

- **Transfermarkt is never scraped.** Its Terms of Use §11.1 prohibit automated
  access *and* separately prohibit using the content to train machine-learning
  models. Labels come from the CC0-licensed Kaggle mirror
  `davidcariboo/player-scores` instead.
- **There is no entity-resolution problem.** `appearances.csv` ships in the same
  CC0 download as the labels and shares its `player_id`, so features and labels
  join on an integer key. FBref remains available as optional enrichment.
- **Temporal evaluation is the headline.** A measured spike
  ([`plans/01-feasibility-spike.md`](plans/01-feasibility-spike.md)) put EUR MAE
  at €2.37M under a group split and €3.96M under a temporal one. The temporal
  number is the one that reflects deployment, so it is the one reported.

## Setup

```bash
make setup
```

**macOS prerequisite.** LightGBM and XGBoost both link `@rpath/libomp.dylib` and
bundle no copy of it, so without this they fail at import with
`OSError: dlopen ... Library not loaded`:

```bash
brew install libomp
```

scikit-learn bundles its own OpenMP, which is why "sklearn works but LightGBM
doesn't" is such a common and confusing report. On Linux and in Docker the
manylinux wheels bundle libgomp and no action is needed.

## Pipeline

Three stages, each reading what the last one wrote. Every figure below is
printed by the command above it, so nothing here is a number someone typed in.

```bash
python scripts/fetch_data.py       # Kaggle -> data/raw -> data/processed (parquet)
python scripts/validate_data.py    # contract checks; --strict fails on warnings too
python scripts/build_features.py   # one row per player-season, labelled and leak-checked
python scripts/train_baseline.py   # baselines across all three splits
python scripts/train_models.py     # nine tuned families, best saved per variant
```

The training table as of the 2026-08-05 dataset refresh:

| | |
|---|---|
| rows | 36,880 |
| players | 17,053 |
| seasons | 2011–2024 |
| rows with a prior-season value | 19,827 |
| leakage findings | 0 |

The label is the first valuation recorded **after** the season's evidence is
complete — an as-of join with a 120-day tolerance, never an equality merge,
because valuations are an irregular on-change event series. Two model variants
come out of the one table: *performance-only* (every row; the useful model, for
scouting) and *with prior value* (19,827 rows; the accurate model, for
tracking). Shipping only the second would be technically true and practically
useless.

## Baselines

Gradient boosting, test-set metrics in EUR, from `scripts/train_baseline.py`.
The **temporal** row is the headline: it is the only split where the model has
never seen the season it is asked about, which is the only situation it will
meet in use.

| Split | performance-only | + prior value |
|---|---|---|
| Random (flattering) | R² 0.499 / MAE €2.84M | R² 0.822 / MAE €2.35M |
| Group by player | R² 0.550 / MAE €2.60M | R² 0.786 / MAE €2.33M |
| **Temporal** | **R² 0.414 / MAE €4.52M** | **R² 0.766 / MAE €3.74M** |

The gap between the random and temporal rows is the point. Reporting the random
number would roughly halve the stated error and answer a question nobody
deploying this will ever ask.

Everything is seeded from one constant; two runs agree exactly, and a test
asserts it. The leakage checks re-run after every split, because splitting is
what creates the chance of a row or a player straddling the boundary.

## The model zoo

Nine families (Linear, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting,
XGBoost, LightGBM, CatBoost), each searched on expanding-window folds inside the
training seasons, selected by validation MAE in EUR, then scored once on the
test seasons. LightGBM wins both variants.

| Variant | Winner | Test MAE | Test R² | vs. Phase 6 baseline |
|---|---|---|---|---|
| performance-only | LightGBM | €4.44M | 0.441 | 0.414 |
| with prior value | LightGBM | €3.71M | 0.775 | 0.766 |

**The zoo barely beats the baseline** — R² moves 0.414 → 0.441 and 0.766 →
0.775. Nine families and a hyperparameter search bought about 0.03 and 0.01.
That is worth stating plainly rather than burying: the signal in this data is
in the features, not in the estimator, and an untuned gradient booster gets
most of the way there. Establishing the baseline first (Phase 6) is what makes
that measurable instead of assumed.

Each run writes a versioned artifact to `models/` — the fitted preprocessing
and estimator as one object, plus metrics, named feature importance, the full
leaderboard and the seed — with a readable JSON sidecar. A test asserts a
reloaded artifact reproduces its recorded metrics exactly. The contract is
[`docs/EXPERIMENT_TRACKING.md`](docs/EXPERIMENT_TRACKING.md).

## Development

```bash
make help          # list every target
make test          # unit tests
make quality       # ruff + black --check + mypy
```

Python 3.13. `pandas` is pinned `<3` deliberately — see the comment in
`requirements.txt`.

## Contributors

This project has a single author, Vansh Tomar. `scripts/hooks/commit-msg`
enforces it: commits carrying a `Co-Authored-By` trailer or an AI-generation
marker are rejected, as are commits authored by anyone else. The hook is
version-controlled and wired via `core.hooksPath`, so it survives a reclone.

## Licence

MIT.
