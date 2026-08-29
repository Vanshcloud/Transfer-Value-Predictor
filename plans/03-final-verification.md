# Phase 13 — final verification (2026-08-29)

Every gate below was run, not reasoned about. Six passed on the first try. Five
found something, and the fix is recorded beside the finding.

## Gates that passed unchanged

| Gate | Result |
|---|---|
| `make quality` — ruff, black, mypy | clean, 90 files, 53 typed modules |
| Full test suite (`pytest`, integration included) | **517 passed** |
| Dashboard: eslint, `tsc --noEmit`, `next build` | clean, 8 routes |
| Docker: both images build | api 2.62 GB, frontend 433 MB |
| Compose stack end to end | `/health` ready, 11 endpoints, 6 pages, live prediction |
| `Co-Authored-By` trailers in history | **0** |
| Authors and committers | Vanshcloud \<vanshwar@gmail.com\> only |
| Removed APIs (7 patterns) | 0 real hits — every match is prose *about* the ban |
| `duckdb` imported outside `src/storage/` | 0 |
| Any code path fetching transfermarkt.com | 0 |
| Unreferenced Python modules | 0 |
| `TODO`/`FIXME`/`XXX`/`HACK` | 0 |
| `npm audit --omit=dev` | 0 vulnerabilities |
| README numbers vs. artifacts on disk | exact match (below) |
| README `make` targets and script paths | all resolve |

README figures re-derived from `data/processed/training_table.parquet` and
`models/*.json`, not copied forward:

```
rows                 36,880 -> 36,880      performance_only  MAE EUR 4.44M  R2 0.441
players              17,053 -> 17,053      with_prior_value  MAE EUR 3.71M  R2 0.775
seasons           2011-2024 -> 2011-2024
rows w/ prior value  19,827 -> 19,827
```

## Findings, and what was done

**1. Two live endpoints were missing from the API contract.**
`GET /players/{player_id}/history` and `GET /features/distribution` had been
shipped, tested and served since Phase 11 with `docs/API_CONTRACT.md` unaware of
them. Documented both, and added `tests/unit/test_api_contract_sync.py`, which
diffs the app's OpenAPI paths against the contract's endpoint table in both
directions. 11 of 11 now match. A document checked only by reading it is checked
once, on the day it is written.

While writing the entry the draft claimed the distribution endpoint returns an
empty payload when no model is loaded. `tests/unit/test_api.py` says it is a
`503`. The document was corrected to the code, not the other way round.

**2. MIT was declared in `pyproject.toml` with no `LICENSE` file.**
A declared licence with no text grants nothing. Added `LICENSE` (MIT, 2026 Vansh
Tomar), a `license` field in `frontend/package.json`, and expanded the README
section to say that the *data* is not MIT — labels are CC0 via the Kaggle mirror,
and Transfermarkt's own ToU §11.1 is why nothing fetches it.

**3. Three runtime dependencies were installed and imported nowhere.**
`beautifulsoup4` and `lxml` were carried for a Transfermarkt scrape the licence
forbids and an FBref one Phase 12 declined; nothing in the pipeline parses HTML,
every source serves CSV or JSON. `plotly` was pinned server-side while the only
Plotly that runs is the npm package in the browser. All three removed from
`requirements.txt`; the API image confirms `bs4` and `lxml` are gone. `plotly`
remains installed because **catboost requires it** — a direct pin would have
claimed this project chose it, and it did not.

**4. `--help` ran the pipeline on three scripts.**
`build_features.py`, `build_reports.py` and `train_baseline.py` had no argument
parser, so `--help` fell through to `main()` and started a full feature build or
training run, and a mistyped flag was silently ignored. Each got a one-line
`argparse.ArgumentParser(description=__doc__).parse_args(argv)`. All six scripts
now print usage and reject unknown flags.

**5. CI had no guard on the project's one legal invariant.**
Added a step to the `removed-apis` job that fails on a transfermarkt.com URL in
code, and on any `bs4`/`lxml` import returning. A reviewer cannot see this rule
being broken by inspection, which is exactly when a grep earns its place.

## Release checklist

Green before tagging. Everything here is a command, not a judgement.

```sh
make quality                 # ruff, black, mypy
pytest                       # 517 tests, integration included
cd frontend && npm run lint && npx tsc --noEmit && npm run build
make docker-build            # both images
docker compose up -d && curl -sf localhost:8000/health && docker compose down -v
git log --format='%(trailers:key=Co-Authored-By)' | grep -c .   # must be 0
git log --format='%an' | sort -u                                # Vanshcloud only
```

The remaining gate is CI on GitHub, which runs the same four jobs on push and
additionally proves the suite is green on a clean checkout with `data/raw/` and
`models/` absent — tests that need a trained model skip rather than fail.
