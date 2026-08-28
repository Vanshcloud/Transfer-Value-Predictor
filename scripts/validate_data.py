#!/usr/bin/env python3
"""Validate the ingested tables.

    python scripts/validate_data.py              # errors fail, warnings report
    python scripts/validate_data.py --strict     # warnings fail too

Exit code 0 when the data satisfies its contract, 1 when it does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.validate import validate  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.validation.report import ValidationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    store = DuckDBParquetStore(settings.paths.processed_dir)

    if not store.list_tables():
        print("no tables found — run scripts/fetch_data.py first", file=sys.stderr)
        return 1

    try:
        report = validate(store, strict=args.strict)
    except ValidationError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(f"\n{report.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
