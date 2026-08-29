#!/usr/bin/env python3
"""Generate the evaluation, explainability and comparison outputs.

    python scripts/build_reports.py

Reads the saved model artifacts — it does not train anything — and writes
self-contained HTML into reports/, plus docs/model_comparison.md and a model
card per variant. Exit code 0 on success, 1 if no model has been trained yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.report import build_reports  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402


def main() -> int:
    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    store = DuckDBParquetStore(settings.paths.processed_dir)

    try:
        bundles = build_reports(
            store,
            settings.split,
            settings.paths.model_dir,
            PROJECT_ROOT / "reports",
            PROJECT_ROOT / "docs",
        )
    except KeyError:
        print("no training table — run scripts/build_features.py first", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    written = [path for bundle in bundles for path in bundle.written]
    print(f"\nwrote {len(written)} file(s):")
    for path in written:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
