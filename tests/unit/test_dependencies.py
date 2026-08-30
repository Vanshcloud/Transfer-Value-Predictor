"""The declaration and the lock must not drift apart.

`requirements.txt` is what a human edits and what `pyproject.toml` publishes.
`requirements-lock.txt` is what CI and the Docker images actually install. Two
files describing the same dependency set diverge the moment one is edited
alone — and the failure is invisible, because both files stay individually
valid. This compares them.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "requirements.txt"
LOCK = ROOT / "requirements-lock.txt"
DEV = ROOT / "requirements-dev.txt"
SERVE = ROOT / "requirements-serve.txt"
SERVE_LOCK = ROOT / "requirements-serve-lock.txt"

REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)\s*(\[[^\]]*\])?\s*(.*)$")


def _entries(path: Path) -> dict[str, str]:
    """Package name (normalised) -> the version specifier beside it."""
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        match = REQUIREMENT.match(line)
        if match:
            found[match.group(1).lower().replace("_", "-")] = match.group(3).strip()
    return found


def test_every_declared_package_is_locked() -> None:
    """A package added to the declaration and not re-locked is installed by
    neither CI nor the image — or worse, floats there while pinned here."""
    missing = sorted(set(_entries(DECLARATION)) - set(_entries(LOCK)))
    assert not missing, f"declared but not locked (re-run uv pip compile): {missing}"


def test_the_lock_pins_every_package_exactly() -> None:
    """A range in a lockfile is a lockfile that does not lock."""
    loose = sorted(name for name, spec in _entries(LOCK).items() if not spec.startswith("=="))
    assert not loose, f"not pinned to an exact version: {loose}"


def test_the_lock_satisfies_the_declared_bounds() -> None:
    """The pin must obey the bound the declaration gives a reason for.

    `pandas<3` is the one that matters: pandas 3 changes the default string
    dtype and makes Copy-on-Write permanent, which is a silent behaviour change
    in a categorical-heavy pipeline. A lock that quietly stepped over it would
    keep every build reproducible and every build wrong.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    locked = _entries(LOCK)
    for line in DECLARATION.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        declared = Requirement(line)
        name = declared.name.lower().replace("_", "-")
        pinned = locked.get(name)
        assert pinned, name
        version = Version(pinned.removeprefix("==").strip())
        assert (
            version in declared.specifier
        ), f"{name} is locked at {version} but declared as {declared.specifier}"


def test_pandas_stays_below_3() -> None:
    """Called out on its own because the comment explaining it is load-bearing."""
    assert _entries(LOCK)["pandas"].startswith("==2.")


def test_dev_installs_the_lock_not_the_declaration() -> None:
    """Otherwise a developer resolves fresh while CI and the images use the
    lock, and 'works on my machine' becomes a real defence."""
    body = DEV.read_text()
    assert "-r requirements-lock.txt" in body
    assert not re.search(r"^-r requirements\.txt\s*$", body, re.MULTILINE)


def test_the_docker_image_installs_the_serve_lock() -> None:
    """The image installs the pinned *serving* set — never the declaration,
    which would float, and never the full lock, which would carry the two
    trainers and their 700 MB for an image whose only command is uvicorn."""
    body = (ROOT / "Dockerfile").read_text()
    assert "requirements-serve-lock.txt" in body
    assert "-r requirements.txt" not in body
    assert "-r requirements-lock.txt" not in body


def test_no_html_parser_is_installed() -> None:
    """Nothing here parses HTML: every source serves CSV or JSON, and the two
    scrapes that would have needed one are forbidden (Transfermarkt) and
    declined (FBref). A parser with nothing to parse is supply-chain surface
    with no upside, so it must not return through a transitive dependency."""
    locked = set(_entries(LOCK))
    assert not locked & {"beautifulsoup4", "bs4", "lxml", "html5lib", "soupsieve"}


def test_pyproject_reads_the_declaration_not_the_lock() -> None:
    """An installed wheel should carry the reasoned ranges, not this machine's
    resolution — pinning a library's transitive tree onto its users is rude."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    files = data["tool"]["setuptools"]["dynamic"]["dependencies"]["file"]
    assert files == ["requirements.txt"]


def test_the_serve_set_is_a_subset_of_the_full_one() -> None:
    """Serving must not need a package training does not declare, or the two
    resolutions describe different projects."""
    extra = sorted(set(_entries(SERVE)) - set(_entries(DECLARATION)))
    assert not extra, f"declared for serving but not for the project: {extra}"


def test_the_serve_set_agrees_with_the_full_set_on_shared_bounds() -> None:
    """A bound with a reason behind it must not be relaxed for the image."""
    full, serve = _entries(DECLARATION), _entries(SERVE)
    for name, spec in serve.items():
        assert (
            spec == full[name]
        ), f"{name} is {spec!r} for serving and {full[name]!r} for the project"


def test_the_serve_lock_pins_exactly_and_omits_the_trainers() -> None:
    """xgboost brings 291 MB of CUDA the inference path never opens; catboost
    269 MB plus plotly. Neither is needed to load a LightGBM artifact."""
    locked = _entries(SERVE_LOCK)
    loose = sorted(n for n, spec in locked.items() if not spec.startswith("=="))
    assert not loose, f"not pinned exactly: {loose}"
    assert not set(locked) & {"xgboost", "catboost", "plotly", "nvidia-nccl-cu12"}
    # The things serving genuinely cannot do without.
    assert {"lightgbm", "shap", "scikit-learn", "fastapi", "duckdb"} <= set(locked)


@pytest.mark.parametrize(
    "path", [DECLARATION, LOCK, DEV, SERVE, SERVE_LOCK, ROOT / "requirements-lint.txt"]
)
def test_every_requirements_file_parses(path: Path) -> None:
    assert _entries(path) or path.read_text().strip()


def test_every_model_family_declares_what_loading_it_needs() -> None:
    """Adding a family to the registry must not silently make the serving image
    unable to load an artifact that family produces.

    The mapping itself lives in tests/integration/test_serving_dependencies.py,
    which checks it against the artifacts actually on disk. This half needs no
    artifacts, so it runs on a clean checkout — which is where a new family is
    most likely to be added without one.
    """
    from src.models.registry import MODEL_REGISTRY
    from tests.integration.test_serving_dependencies import PACKAGE_FOR_FAMILY

    unknown = sorted(set(MODEL_REGISTRY) - set(PACKAGE_FOR_FAMILY))
    assert not unknown, f"add these to PACKAGE_FOR_FAMILY: {unknown}"

    serve = _entries(SERVE)
    for family, package in PACKAGE_FOR_FAMILY.items():
        if package in serve:
            continue
        # Not installed for serving is fine — it just means an artifact from
        # that family could not be loaded by the image, which is what the
        # integration test checks against what is actually shipped.
        assert package in _entries(
            DECLARATION
        ), f"{family} needs {package}, which no requirements file declares"
