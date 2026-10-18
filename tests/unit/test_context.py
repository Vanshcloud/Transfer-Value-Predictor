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
    COMPETITION_LEVEL_DATE_COLUMN,
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    attach_context,
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
                        # Published a fortnight after the season closed.
                        "label_date": pd.Timestamp(f"{season + 1}-07-15"),
                    }
                    for _ in range(50)
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _levels(panel: pd.DataFrame, seasons: range = range(2015, 2023)) -> pd.DataFrame:
        """What the builder passes: each season cut at its 1 July boundary."""
        cutoffs = pd.Series(
            [pd.Timestamp(f"{season + 1}-07-01") for season in seasons], index=list(seasons)
        )
        return competition_strength(panel, target_column="market_value_in_eur", cutoffs=cutoffs)

    def test_a_competition_that_left_the_data_leaves_the_ranking(self) -> None:
        """Without `active`, a competition last played in 2018 kept its old
        level in every later season's ranking population."""
        panel = self._panel(hot_from=2020, hot_value=90_000_000.0)
        gone = panel[panel["primary_competition_id"] == "CALM"].assign(
            primary_competition_id="GONE",
            market_value_in_eur=lambda f: f["market_value_in_eur"] * 3,
        )
        panel = pd.concat([panel, gone[gone["season"] <= 2018]], ignore_index=True)
        seasons = range(2015, 2023)
        cutoffs = pd.Series(
            [pd.Timestamp(f"{season + 1}-07-01") for season in seasons], index=list(seasons)
        )
        active = pd.DataFrame(
            [
                (comp, season)
                for season in seasons
                for comp in ("CALM", "HOT", "GONE")
                if comp != "GONE" or season <= 2018
            ],
            columns=["competition_id", "season"],
        )
        ranked = competition_strength(
            panel, target_column="market_value_in_eur", cutoffs=cutoffs, active=active
        ).set_index(["primary_competition_id", "season"])
        assert ("GONE", 2021) not in ranked.index
        assert len(ranked.xs(2018, level="season")) == 3  # still played in 2018
        # Two competitions in 2021, so the ranks are exactly 0.5 and 1.0.
        assert sorted(ranked.xs(2021, level="season")["competition_tier_rank"]) == [0.5, 1.0]

    def test_a_seasons_own_label_is_not_in_its_own_feature(self) -> None:
        """The whole point. HOT's values explode in 2020; its 2020 *feature*
        must still show the pre-explosion level."""
        out = self._levels(self._panel(hot_from=2020, hot_value=90_000_000.0))
        hot = out[out["primary_competition_id"] == "HOT"].set_index("season")
        level = hot["competition_value_level"]

        assert level[2020] < np.log1p(10_000_000), "2020 feature saw the 2020 jump"
        assert level[2021] > level[2020], "2021 should have absorbed 2020"

    def test_the_first_observed_season_has_no_history(self) -> None:
        out = self._levels(self._panel(hot_from=2020, hot_value=90_000_000.0))
        # Absent rather than present-and-null: a competition with no published
        # history has no level to report, and joins as null downstream.
        assert out.loc[out["season"] == 2015, "competition_value_level"].notna().sum() == 0

    def test_changing_only_the_last_season_cannot_change_earlier_features(self) -> None:
        """A direct causality check: perturb 2022 and nothing before it moves."""
        base = self._levels(self._panel(hot_from=2020, hot_value=90_000_000.0)).set_index(
            ["primary_competition_id", "season"]
        )["competition_value_level"]

        panel = self._panel(hot_from=2020, hot_value=90_000_000.0)
        panel.loc[panel["season"] == 2022, "market_value_in_eur"] *= 50
        perturbed = self._levels(panel).set_index(["primary_competition_id", "season"])[
            "competition_value_level"
        ]

        earlier = base.index.get_level_values("season") <= 2022
        pd.testing.assert_series_equal(base[earlier], perturbed[earlier])

    def test_a_thin_competition_season_does_not_define_a_level(self) -> None:
        """Fewer rows than MIN_ROWS_FOR_STRENGTH is noise, not a measurement."""
        thin = pd.DataFrame(
            [
                {
                    "primary_competition_id": "TINY",
                    "season": s,
                    "market_value_in_eur": 1e6,
                    "label_date": pd.Timestamp(f"{s + 1}-07-15"),
                }
                for s in (2018, 2019)
            ]
        )
        out = self._levels(thin, range(2018, 2021))
        assert out["competition_value_level"].isna().all()

    def test_a_label_published_after_the_cutoff_is_not_used(self) -> None:
        """F2. "Earlier season" is not "already published". With a 365-day
        label window, season s-1 can be labelled after season s has closed;
        on the full panel 8,413 training rows (9.79%) encoded such a label.

        Here HOT's 2019 labels are published on 2020-07-15 except a late batch
        of huge ones published 2021-08-01 — after the 2020 season's cutoff of
        2021-07-01. The 2020 level must not see that batch, and 2021 must."""
        panel = self._panel(hot_from=2100, hot_value=0.0)
        late = (panel["primary_competition_id"] == "HOT") & (panel["season"] == 2019)
        late_rows = panel[late].head(25).index
        panel.loc[late_rows, "market_value_in_eur"] = 500_000_000.0
        panel.loc[late_rows, "label_date"] = pd.Timestamp("2021-08-01")

        hot = self._levels(panel).query("primary_competition_id == 'HOT'").set_index("season")
        # Exactly the mean of season means over what was public on 2021-07-01.
        public = panel[
            (panel["primary_competition_id"] == "HOT")
            & (panel["season"] < 2020)
            & (panel["label_date"] < pd.Timestamp("2021-07-01"))
        ]
        expected = public.groupby("season")["market_value_in_eur"].apply(
            lambda v: np.log1p(v).mean()
        )
        assert hot.loc[2020, "competition_value_level"] == pytest.approx(expected.mean())
        assert hot.loc[2020, COMPETITION_LEVEL_DATE_COLUMN] < pd.Timestamp("2021-07-01")
        assert hot.loc[2021, "competition_value_level"] > hot.loc[2020, "competition_value_level"]

    def test_the_source_date_is_strictly_before_every_cutoff(self) -> None:
        out = self._levels(self._panel(hot_from=2018, hot_value=9_000_000.0))
        dated = out.dropna(subset=["competition_value_level"])
        cutoff = pd.to_datetime((dated["season"] + 1).astype(str) + "-07-01")
        assert (dated[COMPETITION_LEVEL_DATE_COLUMN] < cutoff).all()

    def test_a_season_with_no_labels_of_its_own_still_gets_a_level(self) -> None:
        """F3. The season being played has no labels yet, and the old encoding
        only produced levels for seasons present in the labelled frame — so
        all 8,709 current-season rows carried a null level and rank. A level is
        a function of history, so a season needs no label to receive one."""
        panel = self._panel(hot_from=2020, hot_value=90_000_000.0)
        out = self._levels(panel, range(2015, 2024))  # 2023 has no labelled rows
        current = out[out["season"] == 2023].set_index("primary_competition_id")
        assert set(current.index) == {"CALM", "HOT"}
        assert current["competition_value_level"].notna().all()
        assert current["competition_tier_rank"].notna().all()
        assert (
            current.loc["HOT", "competition_tier_rank"]
            > current.loc["CALM", "competition_tier_rank"]
        )

    def test_the_rank_is_within_season(self) -> None:
        out = self._levels(self._panel(hot_from=2016, hot_value=90_000_000.0))
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
                {
                    "game_id": i,
                    "date": f"2020-09-0{i}",
                    "season": 2020,
                    "competition_type": "domestic_league",
                }
                for i in (1, 2, 3)
            ]
        )
        out = club_season_strength(club_games, games)
        # A running record, one row per match date: the last row carries the
        # whole three-match season, the first only the opening win.
        assert len(out) == 3
        assert out["club_points_per_game"].iloc[-1] == pytest.approx(4 / 3)
        assert out["club_goal_difference_per_game"].iloc[-1] == pytest.approx(1 / 3)
        assert out["club_points_per_game"].iloc[0] == pytest.approx(3.0)
        assert out["club_matches"].tolist() == [1, 2, 3]

    def test_the_record_never_includes_a_match_played_later(self) -> None:
        """The reason this function returns a running record at all.

        A season is named for its August and runs to July, so a match on 20 July
        belongs to it while falling after the 1 July as-of boundary most rows
        use. Averaged over the whole season, that match reached 17.6% of rows
        and postdated the label on 1.22% of them.
        """
        club_games = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "club_id": 7,
                    "own_goals": 0,
                    "opponent_goals": 3,
                    "own_position": 8,
                },
                {
                    "game_id": 2,
                    "club_id": 7,
                    "own_goals": 5,
                    "opponent_goals": 0,
                    "own_position": 1,
                },
            ]
        )
        games = pd.DataFrame(
            [
                {"game_id": 1, "date": "2021-05-01", "season": 2020, "competition_type": "d"},
                # After the 1 July boundary, still season 2020.
                {"game_id": 2, "date": "2021-07-20", "season": 2020, "competition_type": "d"},
            ]
        )
        record = club_season_strength(club_games, games)
        table = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "season": 2020,
                    "primary_club_id": 7,
                    "primary_competition_id": "GB1",
                    "as_of_date": pd.Timestamp("2021-07-01"),
                }
            ]
        )
        out = attach_context(
            table,
            competition_mix=pd.DataFrame(columns=["player_id", "season"]),
            club_strength=record,
            competition_levels=pd.DataFrame(columns=["primary_competition_id", "season"]),
        )
        # The 5-0 on 20 July had not been played on 1 July and must not count.
        assert out["club_matches"].iloc[0] == 1
        assert out["club_points_per_game"].iloc[0] == pytest.approx(0.0)
        assert out["club_record_date"].iloc[0] == pd.Timestamp("2021-05-01")

    def test_a_game_with_no_season_is_dropped_not_guessed(self) -> None:
        club_games = pd.DataFrame(
            [{"game_id": 9, "club_id": 7, "own_goals": 1, "opponent_goals": 0, "own_position": 1}]
        )
        games = pd.DataFrame(
            [{"game_id": 9, "date": "2020-09-01", "season": None, "competition_type": "x"}]
        )
        assert club_season_strength(club_games, games).empty


def test_the_declared_context_columns_are_the_ones_produced() -> None:
    """A name in CONTEXT_NUMERIC that nothing produces becomes a silently
    all-null feature, which is worse than a missing one because it looks fine."""
    assert set(CONTEXT_NUMERIC) & set(CONTEXT_CATEGORICAL) == set()
    assert "competition_value_level" in CONTEXT_NUMERIC
    assert "primary_competition_type" in CONTEXT_CATEGORICAL
