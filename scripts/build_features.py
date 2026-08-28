#!/usr/bin/env python3
"""Build the training table from the ingested raw tables.

    python scripts/build_features.py

Reads players / player_valuations / appearances from the processed store,
builds one row per player-season, runs the leakage stage, and writes
`training_table` back to the store. Exit code 0 on success, 1 on a leak or
missing input.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.features import build_features  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.validation.report import ValidationError  # noqa: E402


def main() -> int:
    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    store = DuckDBParquetStore(settings.paths.processed_dir)

    try:
        report = build_features(
            store,
            season_start_month=settings.data.season_start_month,
            tolerance_days=settings.data.label_tolerance_days,
        )
    except KeyError as exc:
        print(f"missing input table {exc} — run scripts/fetch_data.py first", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"\nleakage detected, training table not written:\n{exc}", file=sys.stderr)
        return 1

    print(f"\ntraining table built:\n{report.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
