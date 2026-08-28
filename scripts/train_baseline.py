#!/usr/bin/env python3
"""Train the baselines and print the split comparison table.

    python scripts/train_baseline.py

Fits Ridge and gradient boosting against both model variants and all three
splits, re-running the leakage checks after each split. Exit code 0 on success,
1 on a leak or a missing training table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.train import render_comparison, run_baselines  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.validation.report import ValidationError  # noqa: E402


def main() -> int:
    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    store = DuckDBParquetStore(settings.paths.processed_dir)

    try:
        results = run_baselines(store, settings.split)
    except KeyError:
        print("no training table — run scripts/build_features.py first", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"\nleakage detected, no metrics produced:\n{exc}", file=sys.stderr)
        return 1

    print(f"\n{render_comparison(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
