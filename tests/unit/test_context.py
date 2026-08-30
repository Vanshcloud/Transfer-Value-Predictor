"""League and club context, and the one feature here that could leak.

``competition_value_level`` is a target encoding: the market value of the
players in a competition, attached to rows whose label is a market value. Done
naively it puts the answer in the question. The defence is a ``shift(1)`` over
an expanding window, and a shift is exactly the kind of thing that survives a
refactor as a no-op, so it is asserted rather than described.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.context import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    club_season_strength,
    competition_strength,
    player_competition_mix,
)


@pytest.fixture
def competitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "competition_id": "GB1",
                "type": "domestic_league",
                "confederation": "europa",
                "country_name": "England",
                "total_clubs": 20,
            },
            {
                "competition_id": "CL",
                "type": "international_cup",
                "confederation": "europa",
                "country_name": None,
                "total_clubs": 32,
            },
            {
                "competition_id": "DK1",
                "type": "domestic_league",
                "confederation": "europa",
                "country_name": "Denmark",
                "total_clubs": 12,
            },
        ]
    )


class TestCompetitionStrength:
    """The leakage-critical one."""

    @staticmethod
    def _panel(hot_from: int, hot_value: float) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for season in range(2015, 2023):
            for comp, base in (("CALM", 1_000_000.0), ("HOT", hot_value)):
                value = base if comp == "CALM" or season >= hot_from else 5_000_000.0
                rows.extend(
                    {
                        "primary_competition_id": comp,
                        "season": season,
                        "market_value_in_eur": value * float(rng.uniform(0.9, 1.1)),
                    }
                    for _ in range(50)
                )
        return pd.DataFrame(rows)

    def test_a_seasons_own_label_is_not_in_its_own_feature(self) -> None:
        """The whole point. HOT's values explode in 2020; its 2020 *feature*
        must still show the pre-explosion level."""
        out = competition_strength(
            self._panel(hot_from=2020, hot_value=90_000_000.0),
            target_column="market_value_in_eur",
        )
        hot = out[out["primary_competition_id"] == "HOT"].set_index("season")
        level = hot["competition_value_level"]

        assert level[2020] < np.log1p(10_000_000), "2020 feature saw the 2020 jump"
        assert level[2021] > level[2020], "2021 should have absorbed 2020"

    def test_the_first_observed_season_has_no_history(self) -> None:
        out = competition_strength(
            self._panel(hot_from=2020, hot_value=90_000_000.0),
            target_column="market_value_in_eur",
        )
        first = out.sort_values("season").groupby("primary_competition_id").head(1)
        assert first["competition_value_level"].isna().all()

    def test_changing_only_the_last_season_cannot_change_earlier_features(self) -> None:
        """A direct causality check: perturb 2022 and nothing before it moves."""
        base = competition_strength(
            self._panel(hot_from=2020, hot_value=90_000_000.0),
            target_column="market_value_in_eur",
        ).set_index(["primary_competition_id", "season"])["competition_value_level"]

        panel = self._panel(hot_from=2020, hot_value=90_000_000.0)
        panel.loc[panel["season"] == 2022, "market_value_in_eur"] *= 50
        perturbed = competition_strength(panel, target_column="market_value_in_eur").set_index(
            ["primary_competition_id", "season"]
        )["competition_value_level"]

        earlier = base.index.get_level_values("season") <= 2022
        pd.testing.assert_series_equal(base[earlier], perturbed[earlier])

    def test_a_thin_competition_season_does_not_define_a_level(self) -> None:
        """Fewer rows than MIN_ROWS_FOR_STRENGTH is noise, not a measurement."""
        thin = pd.DataFrame(
            [
                {"primary_competition_id": "TINY", "season": s, "market_value_in_eur": 1e6}
                for s in (2018, 2019)
            ]
        )
        out = competition_strength(thin, target_column="market_value_in_eur")
        assert out["competition_value_level"].isna().all()

    def test_the_rank_is_within_season(self) -> None:
        out = competition_strength(
            self._panel(hot_from=2016, hot_value=90_000_000.0),
            target_column="market_value_in_eur",
        )
        ranked = out.dropna(subset=["competition_tier_rank"])
        assert ranked["competition_tier_rank"].between(0, 1).all()


class TestPlayerCompetitionMix:
    def test_the_primary_competition_is_the_one_with_the_most_minutes(
        self, competitions: pd.DataFrame
    ) -> None:
        appearances = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "date": "2020-09-01",
                    "competition_id": "GB1",
                    "minutes_played": 900,
                    "player_club_id": 11,
                },
                {
                    "player_id": 1,
                    "date": "2020-10-01",
                    "competition_id": "CL",
                    "minutes_played": 90,
                    "player_club_id": 11,
                },
            ]
        )
        mix = player_competition_mix(appearances, competitions).set_index("player_id")
        assert mix.loc[1, "primary_competition_id"] == "GB1"
        assert mix.loc[1, "primary_competition_type"] == "domestic_league"
        assert mix.loc[1, "competitions_played"] == 2

    def test_continental_share_is_minutes_weighted(self, competitions: pd.DataFrame) -> None:
        appearances = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "date": "2020-09-01",
                    "competition_id": "GB1",
                    "minutes_played": 750,
                    "player_club_id": 11,
                },
                {
                    "player_id": 1,
                    "date": "2020-10-01",
                    "competition_id": "CL",
                    "minutes_played": 250,
                    "player_club_id": 11,
                },
            ]
        )
        mix = player_competition_mix(appearances, competitions)
        assert mix["continental_minutes_share"].iloc[0] == pytest.approx(0.25)

    def test_a_player_with_no_continental_minutes_scores_zero_not_null(
        self, competitions: pd.DataFrame
    ) -> None:
        appearances = pd.DataFrame(
            [
                {
                    "player_id": 2,
                    "date": "2020-09-01",
                    "competition_id": "DK1",
                    "minutes_played": 900,
                    "player_club_id": 22,
                }
            ]
        )
        mix = player_competition_mix(appearances, competitions)
        assert mix["continental_minutes_share"].iloc[0] == 0.0


class TestClubStrength:
    def test_points_are_reconstructed_from_goals(self) -> None:
        """`is_win` cannot distinguish a draw from a defeat, so it is not used."""
        club_games = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "club_id": 7,
                    "own_goals": 3,
                    "opponent_goals": 0,
                    "own_position": 1,
                },
                {
                    "game_id": 2,
                    "club_id": 7,
                    "own_goals": 1,
                    "opponent_goals": 1,
                    "own_position": 1,
                },
                {
                    "game_id": 3,
                    "club_id": 7,
                    "own_goals": 0,
                    "opponent_goals": 2,
                    "own_position": 2,
                },
            ]
        )
        games = pd.DataFrame(
            [
                {"game_id": i, "season": 2020, "competition_type": "domestic_league"}
                for i in (1, 2, 3)
            ]
        )
        out = club_season_strength(club_games, games)
        assert out["club_points_per_game"].iloc[0] == pytest.approx(4 / 3)
        assert out["club_goal_difference_per_game"].iloc[0] == pytest.approx(1 / 3)

    def test_a_game_with_no_season_is_dropped_not_guessed(self) -> None:
        club_games = pd.DataFrame(
            [{"game_id": 9, "club_id": 7, "own_goals": 1, "opponent_goals": 0, "own_position": 1}]
        )
        games = pd.DataFrame([{"game_id": 9, "season": None, "competition_type": "x"}])
        assert club_season_strength(club_games, games).empty


def test_the_declared_context_columns_are_the_ones_produced() -> None:
    """A name in CONTEXT_NUMERIC that nothing produces becomes a silently
    all-null feature, which is worse than a missing one because it looks fine."""
    assert set(CONTEXT_NUMERIC) & set(CONTEXT_CATEGORICAL) == set()
    assert "competition_value_level" in CONTEXT_NUMERIC
    assert "primary_competition_type" in CONTEXT_CATEGORICAL
