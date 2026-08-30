"""The richer performance features, and the arithmetic behind each one."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.performance import (
    PERFORMANCE_NUMERIC,
    attach_performance,
    match_level_features,
    squad_match_counts,
    squad_role,
)


@pytest.fixture
def appearances() -> pd.DataFrame:
    """One player, one season: four matches, goals concentrated late."""
    return pd.DataFrame(
        [
            {"player_id": 1, "game_id": 1, "date": "2020-08-15", "minutes_played": 90, "goals": 0},
            {"player_id": 1, "game_id": 2, "date": "2020-10-15", "minutes_played": 20, "goals": 0},
            {"player_id": 1, "game_id": 3, "date": "2021-03-15", "minutes_played": 90, "goals": 2},
            {"player_id": 1, "game_id": 4, "date": "2021-05-15", "minutes_played": 75, "goals": 1},
        ]
    )


class TestMatchLevelFeatures:
    def test_full_match_share_counts_real_involvements(self, appearances: pd.DataFrame) -> None:
        out = match_level_features(appearances)
        # 90, 90 and 75 clear the 60-minute line; the 20-minute cameo does not.
        assert out["full_match_share"].iloc[0] == pytest.approx(0.75)

    def test_scoring_match_share(self, appearances: pd.DataFrame) -> None:
        out = match_level_features(appearances)
        assert out["scoring_match_share"].iloc[0] == pytest.approx(0.5)

    def test_second_half_goal_share_detects_a_late_run(self, appearances: pd.DataFrame) -> None:
        """All three goals came after the midpoint of this player's season."""
        out = match_level_features(appearances)
        assert out["second_half_goal_share"].iloc[0] == pytest.approx(1.0)

    def test_a_goalless_season_has_no_share_rather_than_zero(self) -> None:
        """Undefined, not zero: a defender who never scored has no distribution
        of goals to be early or late, and 0.0 would assert he scored early."""
        goalless = pd.DataFrame(
            [
                {
                    "player_id": 2,
                    "game_id": i,
                    "date": f"2020-0{i}-15",
                    "minutes_played": 90,
                    "goals": 0,
                }
                for i in (8, 9)
            ]
        )
        out = match_level_features(goalless)
        assert pd.isna(out["second_half_goal_share"].iloc[0])

    def test_months_active_is_a_rhythm_proxy(self, appearances: pd.DataFrame) -> None:
        out = match_level_features(appearances)
        assert out["months_active"].iloc[0] == 4


class TestSquadRole:
    def test_starts_and_substitutes_are_separated(self) -> None:
        lineups = pd.DataFrame(
            [
                {
                    "date": "2020-09-01",
                    "player_id": 1,
                    "type": "starting_lineup",
                    "position": "Centre-Back",
                    "team_captain": 1,
                },
                {
                    "date": "2020-09-08",
                    "player_id": 1,
                    "type": "starting_lineup",
                    "position": "Left-Back",
                    "team_captain": 0,
                },
                {
                    "date": "2020-09-15",
                    "player_id": 1,
                    "type": "substitutes",
                    "position": "Centre-Back",
                    "team_captain": 0,
                },
            ]
        )
        out = squad_role(lineups).iloc[0]
        assert out["starts"] == 2
        assert out["substitute_appearances"] == 1
        assert out["start_share"] == pytest.approx(2 / 3)
        assert out["captain_share"] == pytest.approx(1 / 3)
        assert out["positions_played"] == 2

    def test_seasons_before_the_lineup_data_produce_nothing(self) -> None:
        """game_lineups.csv starts in 2013. Earlier seasons must join to null,
        never to zero: 'did not start' and 'unknown' are different claims."""
        assert squad_role(
            pd.DataFrame(columns=["date", "player_id", "type", "position", "team_captain"])
        ).empty


class TestAvailability:
    def test_squad_match_share_uses_the_clubs_own_fixture_count(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "season": 2020,
                    "primary_club_id": 7,
                    "appearances": 19,
                    "goals": 4,
                    "assists": 2,
                    "minutes_played": 1500,
                }
            ]
        )
        club_games = pd.DataFrame([{"game_id": i, "club_id": 7} for i in range(38)])
        games = pd.DataFrame([{"game_id": i, "season": 2020} for i in range(38)])
        out = attach_performance(
            table,
            match_features=pd.DataFrame(columns=["player_id", "season"]),
            role=pd.DataFrame(columns=["player_id", "season"]),
            squad_matches=squad_match_counts(club_games, games),
        )
        assert out["squad_match_share"].iloc[0] == pytest.approx(0.5)

    def test_a_midseason_move_cannot_push_availability_above_one(self) -> None:
        """A player can appear for two clubs, so his match count may exceed
        either club's total. Above 1 the ratio stops meaning availability."""
        table = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "season": 2020,
                    "primary_club_id": 7,
                    "appearances": 50,
                    "goals": 0,
                    "assists": 0,
                    "minutes_played": 4000,
                }
            ]
        )
        club_games = pd.DataFrame([{"game_id": i, "club_id": 7} for i in range(30)])
        games = pd.DataFrame([{"game_id": i, "season": 2020} for i in range(30)])
        out = attach_performance(
            table,
            match_features=pd.DataFrame(columns=["player_id", "season"]),
            role=pd.DataFrame(columns=["player_id", "season"]),
            squad_matches=squad_match_counts(club_games, games),
        )
        assert out["squad_match_share"].iloc[0] == 1.0

    def test_goal_contributions_combine_goals_and_assists(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "season": 2020,
                    "primary_club_id": 7,
                    "appearances": 30,
                    "goals": 12,
                    "assists": 8,
                    "minutes_played": 2700,
                }
            ]
        )
        out = attach_performance(
            table,
            match_features=pd.DataFrame(columns=["player_id", "season"]),
            role=pd.DataFrame(columns=["player_id", "season"]),
            squad_matches=pd.DataFrame(columns=["club_id", "season", "club_season_matches"]),
        )
        assert out["goal_contributions"].iloc[0] == 20
        assert out["contributions_per_90"].iloc[0] == pytest.approx(20 * 90 / 2700)


def test_every_declared_performance_feature_is_produced() -> None:
    """A declared name nothing produces becomes an all-null column that looks
    like a feature and is not one."""
    assert len(set(PERFORMANCE_NUMERIC)) == len(PERFORMANCE_NUMERIC)
    assert "squad_match_share" in PERFORMANCE_NUMERIC
    assert "start_share" in PERFORMANCE_NUMERIC
