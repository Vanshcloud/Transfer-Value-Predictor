"""One version, declared once per ecosystem, and nothing allowed to disagree.

Phase 13's audit found `0.1.0` in five places while the release tag said
`v1.0.0` — including the version `/health` reports to clients. Copies drift
because nothing compares them, so this compares them.

`src/__init__.py` is the Python source of truth: `pyproject.toml` reads it via
`[tool.setuptools.dynamic]` and `api/main.py` imports it. npm cannot read a
Python file, so `frontend/package.json` keeps its own literal and this test is
what keeps it honest.

The **git tag** is checked here too. It was left out of the first version of
this file, which is a strange omission given that a tag saying v1.0.0 over a
codebase saying 0.1.0 is the exact drift that prompted writing it.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import src
from api.main import VERSION, create_app

ROOT = Path(__file__).resolve().parents[2]


def test_version_is_declared() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", src.__version__)


def test_the_api_reports_the_package_version() -> None:
    assert src.__version__ == VERSION
    assert create_app().openapi()["info"]["version"] == src.__version__


def test_pyproject_does_not_keep_a_second_copy() -> None:
    """A literal `version =` under [project] would shadow the dynamic one."""
    body = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version\s*=\s*"', body, re.MULTILINE) is None
    assert 'version = {attr = "src.__version__"}' in body


def test_the_dashboard_version_matches() -> None:
    package = json.loads((ROOT / "frontend" / "package.json").read_text())
    assert package["version"] == src.__version__


def _git(*args: str) -> str | None:
    """Run a git command, or return None where git or the repo is unavailable.

    A wheel installed from PyPI has no .git directory, and a shallow CI
    checkout has no tags. Neither is a version defect.
    """
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def test_the_latest_tag_matches_the_declared_version() -> None:
    """A tag is a claim about what the code says it is.

    The audit that produced this file found `v1.0.0` tagged over five files
    that all said `0.1.0`, including the version `/health` reports to clients.
    Tags are cheap to get wrong and are read by people who will never open
    src/__init__.py.
    """
    tags = _git("tag", "--list", "v*")
    if not tags:
        pytest.skip("no tags in this checkout (shallow clone, or an installed wheel)")

    versions = sorted(
        (tuple(int(part) for part in tag.lstrip("v").split(".")), tag)
        for tag in tags.splitlines()
        if re.fullmatch(r"v\d+\.\d+\.\d+", tag)
    )
    assert versions, f"tags exist but none is a vX.Y.Z release tag: {tags.splitlines()}"

    _, latest = versions[-1]
    assert latest == f"v{src.__version__}", (
        f"latest release tag is {latest} but the code declares "
        f"{src.__version__}; bump src/__init__.py or move the tag"
    )


def test_the_changelog_documents_the_declared_version() -> None:
    """Shipping a version the changelog has never heard of."""
    body = (ROOT / "CHANGELOG.md").read_text()
    assert f"[{src.__version__}]" in body, f"CHANGELOG.md has no entry for {src.__version__}"
