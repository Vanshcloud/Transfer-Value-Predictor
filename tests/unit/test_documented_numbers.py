"""Numbers the README states must be numbers the repository can produce.

The project's stated standard is that "every figure is printed by the command
above it". The release audit found three that were not: coverage documented as
97% where the cited command prints 89%, a test count of 475 that had moved, and
"the 29 tests for prediction logic" when there were 58. None of them was ever
wrong on the day it was written — they were wrong because prose does not
recompute and nothing compared it to anything.

This compares it. Counts come from pytest's own collector in a subprocess,
because collecting from inside a running session is not reliable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()

# "530/578 tests collected (48 deselected) in 2.73s"
COLLECTED = re.compile(r"(\d+)/(\d+) tests collected \((\d+) deselected\)")


def _collect() -> tuple[int, int, int]:
    """(credential-free, total, integration) test counts, from pytest itself."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-m",
            "not integration",
            "-p",
            "no:cacheprovider",
            "--no-header",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    match = COLLECTED.search(done.stdout)
    if not match:  # pragma: no cover - only if pytest changes its summary line
        pytest.skip(f"could not parse pytest's collection summary:\n{done.stdout[-500:]}")
    unit, total, deselected = (int(group) for group in match.groups())
    return unit, total, deselected


@pytest.fixture(scope="module")
def counts() -> tuple[int, int, int]:
    return _collect()


def test_the_readme_states_the_real_credential_free_count(
    counts: tuple[int, int, int],
) -> None:
    """The headline promise: clone it, run one command, watch it pass."""
    unit, _, deselected = counts
    assert (
        f"{unit} passing tests and {deselected} skips" in README
    ), f"README should say '{unit} passing tests and {deselected} skips'"


def test_the_readme_states_the_real_clean_clone_count(
    counts: tuple[int, int, int],
) -> None:
    unit, _, deselected = counts
    clean_clone = re.search(r"verified on a fresh clone: (\d+)\s*\n?\s*pass", README)
    assert clean_clone, "README no longer states a fresh-clone result"
    assert int(clean_clone.group(1)) == unit
    assert f"the {deselected} integration tests skip" in README


def test_the_readme_states_the_real_full_suite_count(counts: tuple[int, int, int]) -> None:
    _, total, _ = counts
    assert f"{total} pass" in README, f"README should report {total} for the full suite"


def test_the_readme_states_the_real_prediction_service_count() -> None:
    """Cited to argue that prediction logic needs no running server, so the
    number is part of an argument rather than decoration."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "tests/unit/test_prediction_service.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    found = re.search(r"(\d+) tests collected", done.stdout)
    assert found, done.stdout[-500:]
    assert f"the {found.group(1)} tests for" in README


def test_the_readme_states_the_real_frontend_count() -> None:
    """Counted from the vitest config's own include glob, so adding a test file
    cannot leave the README behind."""
    package = json.loads((ROOT / "frontend" / "package.json").read_text())
    assert "test" in package["scripts"], "frontend has no test script"
    stated = re.search(r"npm test\s+#\s*(\d+) tests", README)
    assert stated, "README no longer states a frontend test count"
    files = sorted((ROOT / "frontend" / "src").rglob("*.test.*"))
    assert files, "README claims frontend tests; none exist"
    # Cheap structural check: every test file is reachable by the include glob.
    assert all(path.suffix in {".ts", ".tsx"} for path in files)


def test_the_readme_coverage_table_matches_the_enforced_floors() -> None:
    """The floors are what CI enforces; the table is what a reader believes."""
    makefile = (ROOT / "Makefile").read_text()
    for target, floor in (("test-cov:", 88), ("test-cov-all:", 96)):
        section = makefile.split(target, 1)[1].split("\n\n", 1)[0]
        assert f"--cov-fail-under={floor}" in section, f"{target} floor moved"

    stated = [int(value) for value in re.findall(r"\*\*(\d\d)%\*\*", README)]
    assert stated, "README no longer states coverage percentages"
    for percentage, floor in zip(sorted(stated), [88, 96], strict=True):
        assert (
            percentage >= floor
        ), f"README claims {percentage}% but the enforced floor is only {floor}%"
