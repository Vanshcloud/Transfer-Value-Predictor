"""Shared fixtures.

The suite never touches the network. Everything reads from data/sample/, a
committed slice of the real dataset: ~190 players plus every valuation and
appearance belonging to them, so joins in tests behave like joins in production
rather than returning empty frames.

Phase 15 added slices of the four context tables — competitions, games,
club_games and game_lineups — restricted to the games and players already in
the sample. Without them 22 of the 41 features were entirely null on this path,
which meant CI type-checked the context and performance code and never once
ran it. With them every feature is populated and every code path executes.

The club strength these produce is a fragment of a real season rather than a
faithful league table, which is correct for a fixture: the point is that the
arithmetic runs on real shapes, not that a club's actual points-per-game is
reproduced from a few hundred rows.
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


@pytest.fixture(scope="session")
def sample_competitions() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "competitions.csv")


@pytest.fixture(scope="session")
def sample_games() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "games.csv", low_memory=False)


@pytest.fixture(scope="session")
def sample_club_games() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "club_games.csv", low_memory=False)


@pytest.fixture(scope="session")
def sample_lineups() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "game_lineups.csv", low_memory=False)


@pytest.fixture(scope="session")
def sample_context(
    sample_competitions: pd.DataFrame,
    sample_games: pd.DataFrame,
    sample_club_games: pd.DataFrame,
    sample_lineups: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Every optional table, keyed as ``build_training_table`` expects."""
    return {
        "competitions": sample_competitions,
        "games": sample_games,
        "club_games": sample_club_games,
        "lineups": sample_lineups,
    }
