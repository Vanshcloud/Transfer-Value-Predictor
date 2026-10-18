"""A prospective test: predictions frozen before their outcomes exist in the data.

Every metric this project reported up to Phase 20 is **retrospective**: the
2023+ test seasons were inspected repeatedly while the model was being built,
and at least three design decisions were informed by them (docs/EVALUATION.md).
No amount of retraining makes them independent again.

What can be independent is a season whose labels nobody working on the model
has seen. The local snapshot's newest valuation is dated before the current
season's as-of date, so every label of that season lies in the future of the
data. :func:`freeze` predicts those rows with the shipped models and records a
hash of the result; committing that hash fixes the predictions in public
before any outcome is downloaded. :func:`score`, run later against a newer
valuations file, rebuilds the labels with the training rule and scores only
valuations published after the freeze cutoff.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate
from src.feature_engineering.build import AS_OF_COLUMN, PRIOR_VALUE_FEATURES
from src.models.artifact import ModelArtifact, predict
from src.models.calibration import interval_for
from src.models.provenance import sha256_file, source_revision

PREDICTIONS_FILE = "predictions.csv"
FREEZE_FILE = "freeze.json"
COLUMNS = [
    "variant",
    "player_id",
    "season",
    AS_OF_COLUMN,
    "prediction_eur",
    "lower_eur",
    "upper_eur",
]


def freeze(
    artifacts: dict[str, tuple[ModelArtifact, str]],
    current: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    tolerance_days: int,
    directory: Path,
    inputs: dict[str, str],
) -> dict[str, Any]:
    """Predict every row whose label cannot yet be in the data, and record it.

    Args:
        artifacts: variant -> (artifact, its SHA-256 from the manifest).
        current: the unlabelled current-season table.
        cutoff: the newest valuation date in the snapshot the models saw.
        tolerance_days: the label window, as in training.
        directory: where to write; must not already hold a freeze.
        inputs: SHA-256 of each input table, recorded for the audit trail.

    Raises:
        FileExistsError: rather than overwrite an earlier freeze.
        ValueError: if a row's as-of date is not after the cutoff, since its
            label could then already be in the data.
    """
    if (directory / FREEZE_FILE).exists():
        raise FileExistsError(f"{directory} already holds a freeze; a freeze is never rewritten")
    rows = current[current["season"] == current["season"].max()].copy()
    rows[AS_OF_COLUMN] = pd.to_datetime(rows[AS_OF_COLUMN])
    if (rows[AS_OF_COLUMN] <= cutoff).any():
        raise ValueError("some rows' labels could already be in the snapshot; not prospective")

    frames = []
    for variant, (artifact, _) in sorted(artifacts.items()):
        usable = rows
        required = [c for c in PRIOR_VALUE_FEATURES if c in artifact.feature_columns]
        if required:
            usable = usable.dropna(subset=required)
        predicted = predict(artifact, usable)
        bounds = [interval_for(artifact.calibration, float(value)) for value in predicted]
        frames.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "player_id": usable["player_id"].to_numpy(),
                    "season": usable["season"].to_numpy(),
                    AS_OF_COLUMN: usable[AS_OF_COLUMN].dt.strftime("%Y-%m-%d").to_numpy(),
                    "prediction_eur": np.round(predicted, 2),
                    "lower_eur": [round(b.get("lower_eur", np.nan), 2) for b in bounds],
                    "upper_eur": [round(b.get("upper_eur", np.nan), 2) for b in bounds],
                }
            )
        )
    table = pd.concat(frames, ignore_index=True).loc[:, COLUMNS]

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PREDICTIONS_FILE
    table.to_csv(path, index=False)
    record = {
        "created_at": datetime.now(UTC).isoformat(),
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "label_rule": (
            f"first valuation on or after the row's as-of date, within {tolerance_days} "
            "days, published after the cutoff"
        ),
        "tolerance_days": tolerance_days,
        "models": {variant: sha for variant, (_, sha) in sorted(artifacts.items())},
        "inputs": inputs,
        "source_revision": source_revision(),
        "predictions_file": PREDICTIONS_FILE,
        "predictions_sha256": sha256_file(path),
        "rows": table.groupby("variant").size().to_dict(),
    }
    (directory / FREEZE_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def score(directory: Path, valuations: pd.DataFrame) -> dict[str, Any]:
    """Score a freeze against valuations published since.

    Only valuations dated after the freeze cutoff are read, so nothing the
    models could have seen enters a label. A row whose window is still open
    and has no valuation yet is *pending*, not wrong; while any are pending
    the scored rows are the early revaluations, which are not a random
    subset, and the result says so.

    Raises:
        ValueError: if the predictions no longer match the hash in the freeze.
    """
    record = json.loads((directory / FREEZE_FILE).read_text(encoding="utf-8"))
    path = directory / record["predictions_file"]
    if sha256_file(path) != record["predictions_sha256"]:
        raise ValueError(f"{path} does not match the hash recorded when it was frozen")

    predictions = pd.read_csv(path, parse_dates=[AS_OF_COLUMN])
    cutoff = pd.Timestamp(record["cutoff"])
    window = pd.Timedelta(int(record["tolerance_days"]), "D")

    later = valuations.loc[:, ["player_id", "date", "market_value_in_eur"]].copy()
    later["date"] = pd.to_datetime(later["date"])
    later = later[later["date"] > cutoff].sort_values("date")
    newest = later["date"].max() if not later.empty else cutoff

    keys = predictions.loc[:, ["player_id", AS_OF_COLUMN]].drop_duplicates()
    labelled = pd.merge_asof(
        keys.sort_values(AS_OF_COLUMN),
        later.rename(columns={"date": "label_date"}),
        left_on=AS_OF_COLUMN,
        right_on="label_date",
        by="player_id",
        direction="forward",
        tolerance=window,
    )
    scored = predictions.merge(labelled, on=["player_id", AS_OF_COLUMN], how="left")
    closed = scored[AS_OF_COLUMN] + window <= newest

    results: dict[str, Any] = {
        "freeze": str(directory),
        "cutoff": record["cutoff"],
        "newest_valuation": newest.strftime("%Y-%m-%d"),
        "variants": {},
    }
    for variant, group in scored.groupby("variant"):
        have = group["market_value_in_eur"].notna()
        actual = group.loc[have, "market_value_in_eur"].to_numpy(float)
        entry: dict[str, Any] = {
            "rows": int(len(group)),
            "labelled": int(have.sum()),
            "pending": int((~have & ~closed.loc[group.index]).sum()),
            "never_labelled": int((~have & closed.loc[group.index]).sum()),
            "complete": bool(closed.loc[group.index].all()),
        }
        if have.any():
            metrics = evaluate(actual, group.loc[have, "prediction_eur"].to_numpy(float))
            inside = (actual >= group.loc[have, "lower_eur"]) & (
                actual <= group.loc[have, "upper_eur"]
            )
            entry.update(
                mae_eur=metrics.mae,
                r2=metrics.r2,
                mape=metrics.mape,
                interval_coverage=float(inside.mean()),
            )
        results["variants"][str(variant)] = entry
    return results


def table_sha256(frame: pd.DataFrame) -> str:
    """A content hash of a frame, independent of how it was stored."""
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


__all__ = ["FREEZE_FILE", "PREDICTIONS_FILE", "freeze", "score", "table_sha256"]
