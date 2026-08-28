#!/usr/bin/env python3
"""Train the model zoo, select a winner per variant, save versioned artifacts.

    python scripts/train_models.py
    python scripts/train_models.py --models ridge lightgbm   # a quick subset

Searches each family on expanding-window folds inside the training seasons,
selects by validation MAE in EUR, then touches the test seasons once. Writes
one joblib plus a readable JSON sidecar per variant into the model directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import MODEL_REGISTRY  # noqa: E402
from src.pipelines.tune import render_leaderboard, run_zoo  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.validation.report import ValidationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REGISTRY),
        help="train only these families (default: all nine)",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    store = DuckDBParquetStore(settings.paths.processed_dir)

    try:
        artifacts = run_zoo(
            store,
            settings.split,
            settings.paths.model_dir,
            model_names=tuple(args.models) if args.models else None,
        )
    except KeyError:
        print("no training table — run scripts/build_features.py first", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"\nleakage detected, no model selected:\n{exc}", file=sys.stderr)
        return 1

    for artifact in artifacts:
        print(f"\n{render_leaderboard(artifact)}")
        print(f"\n{artifact.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
