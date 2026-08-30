"""Rows for the season being played: features complete, label not yet published.

The audit's sixth limitation was that the freshest predictable season was a
year old, because a row could not exist until a valuation existed to label it.
Prediction needs no label. These rows exist so the service can price the season
in progress, and the tests below are mostly about the two ways that could go
wrong: training on them, or letting a "prior" value come from the present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.build import (
    AS_OF_COLUMN,
    FEATURE_COLUMNS,
    LABEL_TIME_COLUMN,
    TARGET_COLUMN,
    build_current_season_table,
    build_training_table,
)


@pytest.fixture(scope="module")
def sources() -> dict[str, pd.DataFrame]:
    """Two players across three seasons; the last one has no valuation yet."""
    appearances = pd.DataFrame(
        [
            {
                "player_id": pid,
                "appearance_id": f"{pid}-{season}-{i}",
                "game_id": i,
                "date": f"{season + 1}-03-1{i}",
                "competition_id": "GB1",
                "player_club_id": 7,
                "minutes_played": 90,
                "goals": 1,
                "assists": 0,
                "yellow_cards": 0,
                "red_cards": 0,
            }
            for pid in (1, 2)
            for season in (2022, 2023, 2024)
            for i in (1, 2, 3)
        ]
    )
    valuations = pd.DataFrame(
        [
            {"player_id": pid, "date": f"{season + 1}-08-01", "market_value_in_eur": 1_000_000.0}
            for pid in (1, 2)
            for season in (2022, 2023)
        ]
    )
    players = pd.DataFrame(
        [
            {
                "player_id": pid,
                "date_of_birth": "1998-01-01",
                "position": "Attack",
                "sub_position": "Centre-Forward",
                "foot": "right",
                "height_in_cm": 180.0,
                "country_of_citizenship": "Brazil",
            }
            for pid in (1, 2)
        ]
    )
    return {"players": players, "valuations": valuations, "appearances": appearances}


@pytest.fixture(scope="module")
def training(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_training_table(sources["players"], sources["valuations"], sources["appearances"])


@pytest.fixture(scope="module")
def current(sources: dict[str, pd.DataFrame], training: pd.DataFrame) -> pd.DataFrame:
    return build_current_season_table(
        sources["players"], sources["valuations"], sources["appearances"], training
    )


class TestCurrentSeasonTable:
    def test_it_holds_only_seasons_after_the_last_labelled_one(
        self, current: pd.DataFrame, training: pd.DataFrame
    ) -> None:
        assert not current.empty
        assert current["season"].min() > training["season"].max()

    def test_the_target_is_absent_not_zero(self, current: pd.DataFrame) -> None:
        """A fit on these rows must fail loudly, not learn that they are free."""
        assert current[TARGET_COLUMN].isna().all()
        assert current[LABEL_TIME_COLUMN].isna().all()

    def test_every_model_feature_is_present(self, current: pd.DataFrame) -> None:
        """The whole point: the model can score these rows unchanged."""
        missing = [c for c in FEATURE_COLUMNS if c not in current.columns]
        assert not missing, missing

    def test_the_prior_value_is_strictly_earlier_than_the_as_of_date(
        self, current: pd.DataFrame
    ) -> None:
        """The one real leakage risk here. A 'previous' valuation dated on or
        after the as-of date would be present information wearing a lag."""
        priced = current.dropna(subset=["prev_value_age_days"])
        assert not priced.empty
        assert (priced["prev_value_age_days"] > 0).all()

    def test_the_prior_value_is_the_most_recent_one_available(
        self, current: pd.DataFrame, sources: dict[str, pd.DataFrame]
    ) -> None:
        """Taken from the raw valuations rather than the last labelled season,
        because a player may have been revalued since — and a fresher prior is
        a better one."""
        row = current[current["player_id"] == 1].iloc[0]
        latest = pd.to_datetime(sources["valuations"]["date"]).max()
        expected_age = (row[AS_OF_COLUMN] - latest).days
        assert row["prev_value_age_days"] == pytest.approx(expected_age)

    def test_one_row_per_player_season(self, current: pd.DataFrame) -> None:
        assert not current.duplicated(subset=["player_id", "season"]).any()

    def test_seasons_observed_counts_the_labelled_history(self, current: pd.DataFrame) -> None:
        """Two labelled seasons behind each player in this fixture."""
        assert (current["seasons_observed"] == 2).all()

    def test_it_is_empty_when_every_season_is_already_labelled(
        self, sources: dict[str, pd.DataFrame], training: pd.DataFrame
    ) -> None:
        ahead = training.assign(season=training["season"] + 100)
        out = build_current_season_table(
            sources["players"], sources["valuations"], sources["appearances"], ahead
        )
        assert out.empty

    def test_a_partial_season_is_priced_as_less_evidence_not_refused(
        self, current: pd.DataFrame
    ) -> None:
        """Half a season of matches is half a season of evidence. The counting
        features fall, which is the honest signal, and nothing errors."""
        assert (current["appearances"] > 0).all()
        assert np.isfinite(current["minutes_played"]).all()
