# Transfer Value Predictor

Predict the market value (EUR) of professional footballers from performance,
biographical and contextual data — and explain every prediction.

> **Status: Phase 5 of 13.** Ingestion, validation and the training table are
> built. Models, API and dashboard are not. See
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
