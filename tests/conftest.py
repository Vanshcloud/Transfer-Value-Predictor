"""Shared fixtures.

The suite never touches the network. Everything reads from data/sample/, a
committed 1 MB slice of the real dataset: 200 players plus every valuation and
appearance belonging to them, so joins in tests behave like joins in production
rather than returning empty frames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def sample_players() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "players.csv")


@pytest.fixture(scope="session")
def sample_valuations() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "player_valuations.csv")


@pytest.fixture(scope="session")
def sample_appearances() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "appearances.csv")
