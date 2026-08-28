"""Smoke test: the package imports and declares a version."""

import src


def test_version_is_declared() -> None:
    assert src.__version__ == "0.1.0"
