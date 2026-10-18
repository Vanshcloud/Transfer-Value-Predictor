# Phase 20 — Codex audit remediation: the remaining findings (F5, F8–F20)

Phase 19 (`plans/09`) fixed F1–F4, F6 and F7 and corrected the selection policy.
This phase reproduces, fixes or documents the rest. The original audit report was
not available in the repository or in the request, so each finding was
reproduced independently from its description before anything changed.

Starting commit `32ab0b1`. Every commit is single-author; no co-author trailers.

## Evidence preserved

| Archive (local, git-ignored) | Contents |
|---|---|
| `archive/pre-remediation-2026-10-18/` | The tables, models and reports before Phase 19 (unchanged) |
| `archive/phase19-first-retrain-2026-10-18/` | Phase 19's first retrain, incl. the Random Forest (unchanged) |
| `archive/phase19-final-2026-10-18/` | The Phase 19 final LightGBM models, tables and model cards, with `SHA256SUMS` and `SOURCE_COMMIT` — archived before this phase rebuilt anything |
| `archive/prospective-2026-10-18/` | The frozen season-2025 predictions (SHA-256 `cd71dbd3…348c1`) |

## Findings

| | Finding | Status | Commit(s) |
|---|---|---|---|
| F5 | Evaluation contamination | **Partially addressed** — documented and a prospective test prepared; no independent assessment completed | `aa71b80`, `250b85c`, `c7d590b` |
| F8 | API numeric validation | **Fixed** | `d722a2f` |
| F9 | Interval interpretation | **Fixed** (presentation, subgroups, recalibration rule); coverage evidence retrospective | `f05da42` |
| F10 | Artifact provenance | **Fixed**; fresh-download reproducibility not demonstrated and not claimed | `ea01602` |
| F11 | Data licensing | **Documented limitation** — legal status uncertain | `e730881` |
| F12 | Statistical correctness | **Fixed** for everything from this phase on; earlier p-values annotated, not recomputed | `aa71b80`, `c7d590b` |
| F13 | Frontend races | **Fixed** | `7f21a16` |
| F14 | Cross-season comparison | **Fixed** | `2567ff3` |
| F15 | Transfer-fee mode | **Fixed** by option B — documented as experimental | `70c5750` |
| F16 | Deployment safeguards | **Fixed** in-process; rate limiting, timeouts and TLS documented as the proxy's | `d6055b9`, `43fcb81` |
| F17 | OpenAPI error schemas | **Fixed** | `cdc88e8` |
| F18 | Accessible charts | **Fixed** | `81ce0a7` |
| F19 | Artifact selection by file name | **Fixed** | `ea01602` |
| F20 | Documentation and configuration consistency | **Fixed** for every document listed in the request | `43fcb81`, `0f46056`, and per finding |

### F8 — reproduced, then fixed

Probed through the real app: an integer past float range on any feature (numeric
or categorical) was a **500** (`OverflowError` escaped validation); shares of 7
and −1, 2.5 goals, 10⁹ minutes, 40 months active, a negative prior age, a log
prior value of 10³⁰⁸ and `"position": 5` all returned **200**. Bounds were added
to `PLAUSIBLE_RANGES` and `WHOLE_NUMBER_FEATURES`, each a definition (a share is a
fraction, a season has ≤ 366 days, a match ≤ 150 minutes, a log valuation ≤
log1p(€10bn)), never an observed maximum; inclusive; nothing clamped. Every stored
row was checked against them first: all pass except four raw heights of 17–19 cm
that had reached the training table while the raw validator claimed preprocessing
nulled them — now nulled in the build. Tests: service and API adversarial cases,
and an integration test that every stored row satisfies every bound.

### F9 — diagnosed on test, decided on pre-test seasons

Performance-only 75.1% coverage: 77.7% in 2023 and 73.2% in 2024, worst below €1M
and above €50M; residual median drifting from +0.084 to +0.137. The decision used
a replay inside the pre-test seasons only (fit before origin *s*, calibrate on *s*,
score *s*+1 and *s*+2, origins 2016–2020): 78.4% / 77.3% for performance-only,
79.7% / 80.6% for prior-value. Rule, committed before training: the smallest
quantile level in 0.80–0.95 whose mean simulated coverage reaches 0.80, both
variants. Result: performance-only 0.825, prior-value 0.80. Test coverage after:
**79.1%** (81.2% / 77.5%) and **80.1%** — retrospective, because the 75.1% had been
seen. Responses carry the prediction's band, its coverage and row count; the
dashboard words it as a group's record, not a player's probability.

### F10 / F19 — provenance and the manifest

The service loaded `*.joblib` in file-name order and kept the last per variant;
the report builder took the first. `models/manifest.json` now names one file per
variant with its SHA-256, verified before unpickling; unlisted files are ignored,
no manifest loads nothing, a model whose features the served tables lack is
refused. Artifacts are schema v3 with a provenance block (commit and dirty flag,
package versions, lockfile hashes, raw-file and table hashes, configuration in
force, target definition, selection rule, feature schema). The record states that
the raw snapshot is not reproducible by re-downloading.

### F11 — what is and is not established

The dataset card separates the publisher's CC0 dedication, verified ownership
(none beyond the declaration), Transfermarkt's terms (§11.1 prohibits automated
access and ML training on the content, and reserves text and data mining under
§44b UrhG — quoted from the 2026-08-28 fetch; the site refused an automated fetch
on 2026-10-18) and what the repository does (never contacts Transfermarkt; trains
on the mirror; redistributes no raw data). Status: uncertain, no clearance.
Resolving it needs written permission from Transfermarkt or legal advice.

### F12 — design effects measured

Player-clustered bootstrap (`src/evaluation/inference.py`): pooled over seven
seasons the design effect on MAE is 3.8 (performance-only) and 3.1 (prior-value);
over the two test seasons about 1.5. Earlier row-level p-values are annotated in
`docs/EVALUATION.md` §5. Paired variant comparison on 13,315 shared rows: the
prior value is worth €1.17M of MAE (1.10M–1.24M).

### F13, F14, F17, F18 — reproduced with failing tests first

Each fix landed with tests verified to fail on the previous code: seven race tests
(two unmount checks pass on both, since React 19 no longer warns), the season
warning, a contract test that provokes every declared error status (it failed on
the old declarations), and table-shape tests for charts and the radar.

### F15 — option B

No CLI path ever trained on fees; three documents and the 1.2.0 changelog said
`--target transfer_fee` would. A second target path for 8.5% coverage and a
sale-conditioned label was not justified; the helpers are documented as
experimental and a test fails if the claim returns.

### F16 — evaluated against the intended deployment

A public read-only demonstration; no authentication added. In-process: 64 KB body
limit (chunked bodies counted), inference slots per core with 503 and
`Retry-After`, global SHAP cached, `/ready`, uvicorn connection limits, compose
memory and CPU limits from the measured 710 MB start-up size. Proxy: rate limits,
timeouts, TLS, with the reason each cannot be done correctly in-process. Tested
without load testing; no sustained-traffic claim.

### F20 — settings and claims

Removed settings nothing read (`default_horizon_days`, `min_age`, `max_age`);
made `API_HOST`/`API_PORT` effective through `python -m api`. Corrected, among
others: "no confirmed leakage remained", a 61,522-row prior table, 0.77 coverage,
a stacked model "not shipping" at a cost, "9 of 10 files", "trains nine",
`--target transfer_fee`, test seasons "touched once" and "never saw", and the
experiment-tracking document's fields no artifact carried.

## Carried-over risks

| Risk | Outcome |
|---|---|
| Momentum features without provenance | **Fixed**: `prev_as_of_date` carried and checked per row as strict provenance |
| Stale competition strength | **Fixed** for the rank: 0–2 departed competitions a season sat in the ranking population; ranks moved by a median 0.019 (13,731 rows changed). The level itself still averages all past seasons equally — a design choice, not a leak |
| Rows that may never be labelled | **Documented**: 1,007 season-2024 rows, 607 for players valued nothing since; described accurately, and dropped by the build once their window closes |
| Performance-only undercoverage | **Addressed** by F9 |
| Authorship-guard failures | **Demonstrated environmental** and fixed in the test: exit 69 from the Xcode git shim with `/usr/bin` first, exit 0 with Homebrew's git |
| Newer prior vs better algorithm | **Documented**: the Phase 19 gain is the prior's definition; both variants are the same family |

## New defects introduced, and resolved, in this phase

- The manifest made `models/*.json` match `manifest.json`, breaking four
  integration tests — fixed (`8adf0a3`).
- The serving-performance tests were parametrized by the models on disk, so a
  clean clone collected four fewer tests than the README stated — fixed
  (`8adf0a3`, `8716ad4`).
- A commit (`70c5750`) landed without its test run because a lint failure
  short-circuited the command; the follow-up `b8d7d0f` fixed the lint and the
  tests passed. Not amended, to keep history unrewritten.

## Retraining

Rebuilt and retrained once, after every data-affecting change (heights, tier
rank, momentum provenance) and the F9 rule. Leakage stage: 0 errors, 0 warnings.
LightGBM selected for both variants by the unchanged policy.

| | Validation MAE | Test MAE (95% CI, players) | Test R² | Coverage | Width |
|---|---|---|---|---|---|
| performance-only | €1,892,889 | €2,057,285 (1.96M–2.15M) | 0.841 | 79.1% | €2.23M |
| with prior value | €900,232 | €908,625 (0.86M–0.96M) | 0.963 | 80.1% | €1.01M |

Before (Phase 19 final): €2,079,666 / 75.1% and €912,394 / 80.3%. All test figures
retrospective. Artifacts 2.2 MB each, explained-prediction latency 25 ms and 23 ms.

## Verification

All run on 2026-10-18/19 at the final code, on this machine (macOS, Python 3.13,
Node 24, Docker 29.8):

| Check | Result |
|---|---|
| Backend, full suite with data and models | **892 passed**, 0 failed; coverage **96.87%** (floor 96) |
| Backend, credential-free (`-m "not integration"`) | 825 passed, 67 deselected; coverage **89.88%** (floor 88) |
| Fresh clone, no data, no models, no credentials | 825 passed, 67 skipped, 0 failed |
| ruff, black, mypy | clean |
| Frontend: vitest, tsc, eslint | 118 passed; clean; clean |
| Next.js production build | passes |
| Leakage stage on the rebuilt tables | 0 errors, 0 warnings |
| OpenAPI vs behaviour | every declared error status provoked and schema-validated |
| `docker compose build` | both images build |
| Compose smoke | API healthy, `/ready` 200 with both models, a prediction with its band's coverage, 413 for a 70 KB body, 422 for a 400-digit integer, dashboard 200; API 589 MiB of its 2 GiB limit |
| `pip check` | no broken requirements |
| `pip-audit` | serve lock: none; full lock: 3 advisories in `urllib3` 2.7.0 (fixed in 2.8.0; used only by `fetch_data.py`) |
| `npm audit --omit=dev` | **critical** advisories in `next` 16.3.3 (fix: 16.4.2, outside the pinned range), `maplibre-gl` via `plotly.js`, high in `sharp`. Not fixed: outside the audit's findings and a framework upgrade needing approval. The app uses none of `next/og`, image optimisation, `use cache`, draft mode or ISR revalidation; the static-page cache-poisoning advisories were not assessed in depth |
| Authorship | every Phase 20 commit authored by Vanshcloud, no co-author trailer |
