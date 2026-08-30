"""One version, declared once per ecosystem, and nothing allowed to disagree.

Phase 13's audit found `0.1.0` in five places while the release tag said
`v1.0.0` — including the version `/health` reports to clients. Copies drift
because nothing compares them, so this compares them.

`src/__init__.py` is the Python source of truth: `pyproject.toml` reads it via
`[tool.setuptools.dynamic]` and `api/main.py` imports it. npm cannot read a
Python file, so `frontend/package.json` keeps its own literal and this test is
what keeps it honest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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
