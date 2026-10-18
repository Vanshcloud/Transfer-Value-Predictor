"""One file that describes one trained model completely.

The contract is docs/EXPERIMENT_TRACKING.md: a saved artifact must be
self-describing, so answering "what produced this number?" needs nothing but
the file. That means the fitted preprocessing travels *inside* the pipeline —
a preprocessor fitted separately is the classic way training and serving drift
apart — and every field that would otherwise live in a config the file does not
carry is copied into it.

A JSON sidecar is written beside the joblib so the metadata is readable without
unpickling, which matters when the question is "which run was best?" and the
answer should not require loading every model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import Metrics
from src.models.provenance import sha256_file
from src.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_VERSION = 3
"""Schema version. Bump when a field changes meaning, so an old artifact
loaded by new code fails loudly instead of being silently misread.

v2 adds `calibration`: measured residual quantiles the API serves as a
prediction interval. A v1 artifact has no interval to serve, so refusing it is
correct — the alternative is an endpoint that silently omits a documented
field.

v3 adds `provenance` (source revision, environment, data and configuration
hashes — :mod:`src.models.provenance`), and artifacts are loaded through the
directory's manifest, never by file name."""

MANIFEST_NAME = "manifest.json"
"""Which file serves each variant, and its SHA-256.

Written by :func:`save`, read by :func:`load_manifest`. It replaces "the
lexicographically last ``*.joblib`` per variant", under which a stale or
misleadingly named file decided what the API served."""


@dataclass
class ModelArtifact:
    """A trained model plus everything needed to trust its numbers."""

    variant: str
    model_name: str
    params: dict[str, Any]
    pipeline: Pipeline
    validation: Metrics
    test: Metrics
    feature_columns: tuple[str, ...]
    target_column: str
    feature_importance: dict[str, float]
    dataset: dict[str, Any]
    split: dict[str, Any]
    seed: int
    calibration: dict[str, Any] = field(default_factory=dict)
    """Measured residual quantiles per value band. Empty means the API reports
    no interval rather than a made-up one."""
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    """What the model was made from. See :mod:`src.models.provenance`."""
    artifact_version: int = ARTIFACT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def metadata(self) -> dict[str, Any]:
        """Everything except the fitted pipeline, as plain JSON-able data."""
        return {
            "artifact_version": self.artifact_version,
            "created_at": self.created_at,
            "variant": self.variant,
            "model_name": self.model_name,
            "params": {key: _plain(value) for key, value in self.params.items()},
            "metrics": {
                "validation": _metrics_dict(self.validation),
                "test": _metrics_dict(self.test),
            },
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "feature_importance": self.feature_importance,
            "dataset": self.dataset,
            "split": self.split,
            "calibration": self.calibration,
            "seed": self.seed,
            "leaderboard": self.leaderboard,
            "provenance": self.provenance,
        }

    def render(self) -> str:
        top = sorted(self.feature_importance.items(), key=lambda kv: -abs(kv[1]))[:8]
        importance = "\n".join(f"    {name:<34} {value:.4f}" for name, value in top)
        return "\n".join(
            [
                f"  variant     {self.variant}",
                f"  model       {self.model_name}  {self.params or 'defaults'}",
                f"  validation  {self.validation.render()}",
                f"  test        {self.test.render()}",
                "  top features by importance:",
                importance or "    (none reported)",
            ]
        )


def _plain(value: object) -> object:
    """Unwrap numpy scalars so ``json.dump`` does not choke on them."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _metrics_dict(metrics: Metrics) -> dict[str, float]:
    return {
        "mae_eur": metrics.mae,
        "rmse_eur": metrics.rmse,
        "r2": metrics.r2,
        "mape": metrics.mape,
        "n": metrics.n,
    }


def extract_feature_importance(pipeline: Pipeline) -> dict[str, float]:
    """Named importances, after preprocessing, so they are readable.

    Linear families expose ``coef_`` and tree families ``feature_importances_``;
    a family that exposes neither returns an empty mapping rather than raising,
    because a missing explanation is not a reason to lose a trained model.
    Names come from the preprocessor, which is why it is configured with
    ``set_output(transform="pandas")``.
    """
    estimator = pipeline.named_steps["model"].regressor_
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        coefficients = getattr(estimator, "coef_", None)
        values = np.ravel(coefficients) if coefficients is not None else None
    if values is None:
        return {}

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if len(names) != len(values):  # pragma: no cover - defensive
        return {}
    return {str(name): float(value) for name, value in zip(names, values, strict=True)}


def save(artifact: ModelArtifact, directory: Path) -> Path:
    """Write the artifact and its readable sidecar, and make it the one the
    manifest serves for its variant. Returns the joblib path."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{artifact.variant}__{artifact.model_name}"

    model_path = directory / f"{stem}.joblib"
    joblib.dump(artifact, model_path)
    checksum = sha256_file(model_path)

    sidecar = directory / f"{stem}.json"
    sidecar.write_text(
        json.dumps({**artifact.metadata(), "sha256": checksum}, indent=2), encoding="utf-8"
    )

    manifest = load_manifest(directory) or {}
    manifest[artifact.variant] = {
        "file": model_path.name,
        "sha256": checksum,
        "model_name": artifact.model_name,
        "artifact_version": artifact.artifact_version,
        "created_at": artifact.created_at,
    }
    partial = directory / f"{MANIFEST_NAME}.part"
    partial.write_text(
        json.dumps({"manifest_version": 1, "variants": manifest}, indent=2), encoding="utf-8"
    )
    partial.replace(directory / MANIFEST_NAME)

    logger.info("saved %s (%.1f KB)", model_path.name, model_path.stat().st_size / 1024)
    return model_path


def load_manifest(directory: Path) -> dict[str, dict[str, Any]] | None:
    """Variant -> manifest entry, or None when the directory has no manifest.

    Raises:
        ValueError: if the manifest exists but is malformed, or names a file
            outside the directory.
    """
    path = directory / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        variants = json.loads(path.read_text(encoding="utf-8"))["variants"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not a valid manifest: {exc}") from exc
    for variant, entry in variants.items():
        if not isinstance(entry, dict) or {"file", "sha256"} - set(entry):
            raise ValueError(f"{path}: entry for {variant!r} lacks file or sha256")
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError(f"{path}: {entry['file']!r} is not a file in this directory")
    result: dict[str, dict[str, Any]] = variants
    return result


def load_variant(directory: Path, variant: str) -> ModelArtifact:
    """The artifact the manifest serves for ``variant``, checksum verified.

    Raises:
        FileNotFoundError: if the directory has no manifest entry for it.
        ValueError: on a checksum or schema mismatch, or the wrong variant.
    """
    entry = (load_manifest(directory) or {}).get(variant)
    if entry is None:
        raise FileNotFoundError(
            f"no {variant} artifact in {directory}/{MANIFEST_NAME} — run scripts/train_models.py"
        )
    artifact = load(directory / entry["file"], sha256=entry["sha256"])
    if artifact.variant != variant:
        raise ValueError(f"{entry['file']} holds a {artifact.variant} model, not {variant}")
    return artifact


def load(path: Path, *, sha256: str | None = None) -> ModelArtifact:
    """Load an artifact, refusing one written by an incompatible schema.

    With ``sha256``, the file is hashed and refused on a mismatch *before* it
    is unpickled, so a file swapped in after training is never executed.

    **This unpickles.** ``joblib.load`` executes whatever the file tells it to,
    and the ``artifact_version`` check below runs *after* that has already
    happened — it is a compatibility guard, never a security one. Load only
    artifacts this project produced. The deployment shape assumes exactly that:
    ``docker-compose.yml`` mounts ``models/`` read-only, nothing downloads an
    artifact, and no endpoint accepts one.

    Replacing joblib with a format that cannot execute — ONNX, or the booster's
    own text dump — would remove the assumption rather than document it. It
    would also cost the fitted preprocessing travelling inside the same object,
    which is the property that stops training and serving drifting apart, so it
    is a trade to make deliberately and not a bug to patch.

    Raises:
        ValueError: if the checksum does not match, or the artifact's schema
            version is not the current one.
    """
    if sha256 is not None and sha256_file(path) != sha256:
        raise ValueError(f"{path.name} does not match the checksum its manifest records")
    artifact: ModelArtifact = joblib.load(path)
    if artifact.artifact_version != ARTIFACT_VERSION:
        raise ValueError(
            f"{path.name} has artifact_version {artifact.artifact_version}, "
            f"this code reads version {ARTIFACT_VERSION}"
        )
    return artifact


def predict(artifact: ModelArtifact, frame: pd.DataFrame) -> np.ndarray:
    """Predict in EUR, using exactly the columns the artifact was trained on.

    Going through here rather than touching ``artifact.pipeline`` directly is
    what stops a caller passing columns in a different order, which scikit-learn
    accepts by name but which is worth making impossible anyway.
    """
    predictions: np.ndarray = artifact.pipeline.predict(frame[list(artifact.feature_columns)])
    return predictions
