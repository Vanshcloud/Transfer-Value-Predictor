#!/usr/bin/env python3
"""Freeze predictions for a season whose labels do not yet exist, or score them later.

    python scripts/prospective.py freeze
    python scripts/prospective.py score reports/prospective/<date> path/to/player_valuations.csv

`freeze` predicts the current season with the models the manifest serves and
writes predictions plus a record of every input's hash; commit the printed
hash before downloading any newer data, so the predictions are fixed in
public first. `score` reads a *newer* valuations file — fetched into its own
directory, not over the snapshot the models were trained on — and scores only
valuations published after the freeze cutoff. See docs/EVALUATION.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.prospective import freeze, score  # noqa: E402
from src.models.artifact import load_manifest, load_variant  # noqa: E402
from src.models.provenance import sha256_file  # noqa: E402
from src.pipelines.features import CURRENT_SEASON_TABLE, TRAINING_TABLE  # noqa: E402
from src.storage.duckdb_store import DuckDBParquetStore  # noqa: E402
from src.utils.config import load_settings  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze", help="predict the current season and record the hash")
    scoring = commands.add_parser("score", help="score a freeze against newer valuations")
    scoring.add_argument("freeze_dir", type=Path)
    scoring.add_argument("valuations", type=Path, help="a newer player_valuations.csv")
    args = parser.parse_args(argv)

    if args.command == "score":
        print(json.dumps(score(args.freeze_dir, pd.read_csv(args.valuations)), indent=2))
        return 0

    settings = load_settings()
    store = DuckDBParquetStore(settings.paths.processed_dir)
    manifest = load_manifest(settings.paths.model_dir) or {}
    if not manifest:
        print("no manifest in the model directory — run scripts/train_models.py", file=sys.stderr)
        return 1
    artifacts = {
        variant: (load_variant(settings.paths.model_dir, variant), entry["sha256"])
        for variant, entry in manifest.items()
    }
    cutoff = pd.to_datetime(store.read_table("player_valuations")["date"]).max()
    directory = PROJECT_ROOT / "reports" / "prospective" / datetime.now(UTC).strftime("%Y-%m-%d")
    record = freeze(
        artifacts,
        store.read_table(CURRENT_SEASON_TABLE),
        cutoff=cutoff,
        tolerance_days=settings.data.label_tolerance_days,
        directory=directory,
        inputs={
            name: sha256_file(store.path_for(name))
            for name in (CURRENT_SEASON_TABLE, TRAINING_TABLE, "player_valuations")
        },
    )
    print(json.dumps(record, indent=2))
    print(f"\nwrote {directory}\npredictions sha256 {record['predictions_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
