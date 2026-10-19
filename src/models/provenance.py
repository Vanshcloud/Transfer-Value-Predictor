"""What a trained model was made from, recorded so it can be checked later.

Everything here identifies an input exactly — a file by its SHA-256, the code
by its commit, the environment by its lockfiles and installed versions — and
nothing claims more than that. In particular, these hashes identify *this*
snapshot of the Kaggle dataset; they do not make it re-downloadable. The
dataset is published as a rolling "latest" version, replaced whenever the
publisher's pipeline runs, so a fresh download will normally not match, and the record says which files would
have to be restored for a rebuild to reproduce the table.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

LOCKFILES = ("uv.lock", "requirements-lock.txt", "requirements-serve-lock.txt")

PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "shap",
    "joblib",
    "pyarrow",
)
"""Whatever can change a fitted model or how it is read back."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The commit the code came from, and whether the tree had uncommitted
    changes. ``None`` with the reason when git is unavailable, rather than a
    guess."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("source revision unavailable: %s", exc)
        return {"commit": None, "dirty": None, "unavailable_because": str(exc)[:200]}
    return {"commit": commit, "dirty": bool(status.strip())}


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    lockfiles = {
        name: sha256_file(PROJECT_ROOT / name)
        for name in LOCKFILES
        if (PROJECT_ROOT / name).exists()
    }
    return {"python": platform.python_version(), "packages": versions, "lockfiles": lockfiles}


def files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(paths)
        if path.exists()
    }


def collect(
    *,
    training_table: Path,
    raw_files: Iterable[Path],
    dataset: str,
    configuration: dict[str, Any],
    config_file: Path = PROJECT_ROOT / "configs" / "config.yaml",
) -> dict[str, Any]:
    """The provenance block every artifact carries.

    ``configuration`` is the feature-build and split settings actually in
    force, so a changed default is visible even when the YAML file is not.
    """
    return {
        "source_revision": source_revision(),
        "environment": environment(),
        "raw_data": {
            "dataset": dataset,
            "files": files(raw_files),
            "reproducible_by_download": False,
            "note": (
                "Kaggle serves this dataset as a rolling latest version; restore "
                "these exact files to rebuild the training table."
            ),
        },
        "training_table": {
            "file": training_table.name,
            **files([training_table]).get(training_table.name, {"sha256": None, "bytes": None}),
        },
        "configuration": {
            "file": config_file.name,
            "sha256": sha256_file(config_file) if config_file.exists() else None,
            "in_force": configuration,
        },
    }


__all__ = ["collect", "environment", "files", "sha256_file", "source_revision"]
