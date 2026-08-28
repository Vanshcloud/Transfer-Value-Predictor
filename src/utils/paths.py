"""Filesystem locations, all derived from one anchor.

Every path in this project resolves from :data:`PROJECT_ROOT`. Nothing computes
a location with ``../..`` relative to its own file, because that silently breaks
the moment a module moves and it makes the same code behave differently
depending on the working directory it was invoked from.
"""

from __future__ import annotations

from pathlib import Path

# src/utils/paths.py -> src/utils -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve(path: Path | str) -> Path:
    """Resolve ``path`` against the project root when it is relative.

    An absolute path is returned unchanged, so a deployment can point
    ``DATA_DIR`` at a mounted volume without the code caring.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
