#!/usr/bin/env python3
"""Download the source data and convert it to Parquet.

    python scripts/fetch_data.py            # download if stale, then convert
    python scripts/fetch_data.py --force    # re-download regardless of cache

Requires Kaggle credentials: KAGGLE_USERNAME and KAGGLE_KEY, or a token at
~/.kaggle/kaggle.json. See .env.example.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running a script from scripts/ puts scripts/ on sys.path, not the project
# root, so `import src...` fails. Fixed here rather than by requiring the
# caller to set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.base import SourceUnavailableError  # noqa: E402
from src.ingestion.kaggle_loader import KaggleSource  # noqa: E402
from src.pipelines.ingest import ingest  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.http import HttpClient  # noqa: E402
from src.utils.logging import configure_logging, get_logger  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    logger = get_logger("fetch_data")

    source = KaggleSource(
        dataset=settings.kaggle.dataset,
        files=settings.kaggle.files,
        raw_dir=settings.paths.raw_dir,
        max_age_days=settings.kaggle.max_age_days,
        client=HttpClient(
            timeout_seconds=settings.http.timeout_seconds,
            max_retries=settings.http.max_retries,
            backoff_factor=settings.http.backoff_factor,
            user_agent=settings.http.user_agent,
            min_request_interval_seconds=settings.http.min_request_interval_seconds,
        ),
    )
    store = DuckDBParquetStore(settings.paths.processed_dir)

    try:
        report = ingest(source, store, force=args.force)
    except SourceUnavailableError as exc:
        logger.error("%s", exc)
        return 2

    print(f"\nIngested into {settings.paths.processed_dir}:")
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
