# Project structure

A map of the repository and, more usefully, *why* each boundary exists. Every
line count below is from the working tree; regenerate with
`find src -name '*.py' | xargs wc -l`.

```
.
├── src/                    the library — no HTTP, no CLI, no side effects on import
│   ├── ingestion/          fetch raw CSVs (requests only, never a browser)
│   ├── validation/         schema, quality and leakage checks
│   ├── feature_engineering/ the training table: joins, features, targets
│   ├── models/             registry, splits, tuning, calibration, artifacts
│   ├── pipelines/          the stages that orchestrate the above
│   ├── evaluation/         metrics, error analysis, model cards, reports
│   ├── explainability/     SHAP, global and per-prediction
│   ├── services/           the prediction service the API wraps
│   ├── storage/            DuckDB + Parquet behind a Protocol
│   ├── visualization/      plots for the HTML reports
│   └── utils/              config, logging, HTTP, paths
├── api/                    FastAPI app — routes, schemas, errors, DI
├── frontend/               Next.js dashboard (TypeScript, Tailwind, Recharts)
├── scripts/                the CLI entry points, one per pipeline stage
│   └── hooks/              git hooks, version-controlled
├── tests/
│   ├── unit/               31 files, no data and no credentials needed
│   └── integration/         7 files, skip unless data and models exist
├── configs/config.yaml     committed defaults; .env overrides a subset
├── data/
│   ├── raw/                downloaded CSVs (gitignored)
│   ├── processed/          Parquet tables (gitignored)
│   └── sample/             committed fixture the unit suite runs against
├── models/                 trained artifacts (gitignored)
├── reports/                generated HTML (gitignored; Markdown output is not)
├── docs/                   contracts, cards, tracking, screenshots
├── plans/                  the design and audit record, phase by phase
└── .github/workflows/ci.yml
```

## Where the boundaries are, and why

**`src/` never imports `api/`.** The dependency runs one way. The prediction
service in `src/services/` knows nothing about HTTP, which is what lets a batch
job or a notebook use exactly the code the API uses. CI enforces this rather
than trusting it.

**`scripts/` are thin.** Each is an entry point over a `src/pipelines/` stage —
argument parsing and logging, no logic. If a script grows a decision, the
decision belongs in the pipeline where a test can reach it.

**`src/storage/` is a `Protocol`.** DuckDB is the current implementation and
the only file that names it is `duckdb_store.py`. CI greps for imports that
would leak the engine into the rest of the tree, which is what would make
swapping it a refactor instead of a config change.

**Preprocessing lives inside the model artifact.** `src/models/artifact.py`
serialises the fitted `ColumnTransformer` and the estimator as one object. A
preprocessor fitted separately from its model is the classic way training and
serving drift apart; keeping them in one file makes the drift impossible rather
than discouraged.

**`plans/` is the record, not the docs.** Ten phase documents carrying the
measurements behind each decision — including the ideas that were built,
measured and rejected. `docs/` is what a *user* needs; `plans/` is what a
*reviewer* needs. See [`plans/06`](plans/06-final-research-audit.md) before
proposing a feature.

## The three that are easy to confuse

| Path | Committed? | Regenerate with |
|---|---|---|
| `data/sample/` | **yes** — the unit suite runs on it | never; it is a fixture |
| `data/raw/`, `data/processed/` | no | `scripts/fetch_data.py`, `scripts/build_features.py` |
| `models/` | no | `scripts/train_models.py` |
| `reports/*.html` | no | `scripts/build_reports.py` |
| `docs/MODEL_CARD_*.md`, `docs/model_comparison.md` | **yes** | `scripts/build_reports.py` — generated, so edit the generator |

The model cards are generated from the artifact on disk, which is what stops
them describing a model that is no longer there. Editing them by hand is
undone by the next report build; edit `src/evaluation/model_card.py` instead.

## Reading order

Coming to this cold, the shortest path to understanding it:

1. [`README.md`](README.md) — what it does and why it is shaped this way
2. [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) — what the model is fitted to
3. [`src/feature_engineering/build.py`](src/feature_engineering/build.py) — the
   join everything rests on, and the leakage argument for each step
4. [`src/validation/leakage.py`](src/validation/leakage.py) — the seven checks
   that run as a pipeline stage rather than as tests
5. [`src/pipelines/tune.py`](src/pipelines/tune.py) — how a model is selected,
   and what the explainability constraint costs
6. [`plans/06-final-research-audit.md`](plans/06-final-research-audit.md) — the
   measurements behind the current design, including the rejected ideas
