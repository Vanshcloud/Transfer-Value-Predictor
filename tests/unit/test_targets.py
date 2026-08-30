"""The optional transfer-fee target, and the discipline it inherits."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.build import AS_OF_COLUMN, TARGET_COLUMN
from src.feature_engineering.targets import (
    TARGET_MODES,
    TRANSFER_FEE_COLUMN,
    attach_transfer_fee,
    select_target,
)


@pytest.fixture
def table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "season": [2020, 2020, 2020],
            AS_OF_COLUMN: pd.to_datetime(["2021-07-01"] * 3),
            TARGET_COLUMN: [10_000_000.0, 5_000_000.0, 1_000_000.0],
        }
    )


@pytest.fixture
def transfers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # player 1: a paid move two months after the season. Valid.
            {"player_id": 1, "transfer_date": "2021-09-01", "transfer_fee": 12_000_000},
            # player 2: a free transfer. A zero fee is the absence of a price.
            {"player_id": 2, "transfer_date": "2021-09-01", "transfer_fee": 0},
            # player 3: a paid move BEFORE the as-of date. Must not be used.
            {"player_id": 3, "transfer_date": "2021-01-01", "transfer_fee": 8_000_000},
        ]
    )


class TestAttachTransferFee:
    def test_a_forward_paid_transfer_is_attached(
        self, table: pd.DataFrame, transfers: pd.DataFrame
    ) -> None:
        out = attach_transfer_fee(table, transfers).set_index("player_id")
        assert out.loc[1, TRANSFER_FEE_COLUMN] == 12_000_000

    def test_a_free_transfer_is_not_a_price_of_zero(
        self, table: pd.DataFrame, transfers: pd.DataFrame
    ) -> None:
        """Averaging a free transfer as EUR 0 would teach the model that
        out-of-contract players are worthless."""
        out = attach_transfer_fee(table, transfers).set_index("player_id")
        assert pd.isna(out.loc[2, TRANSFER_FEE_COLUMN])

    def test_a_transfer_before_the_as_of_date_is_never_used(
        self, table: pd.DataFrame, transfers: pd.DataFrame
    ) -> None:
        """Same forward discipline as the market-value label: a move that had
        already happened when the features were observed is not a label, it is
        a leak."""
        out = attach_transfer_fee(table, transfers).set_index("player_id")
        assert pd.isna(out.loc[3, TRANSFER_FEE_COLUMN])

    def test_the_tolerance_bounds_how_long_we_wait(
        self, table: pd.DataFrame, transfers: pd.DataFrame
    ) -> None:
        out = attach_transfer_fee(table, transfers, tolerance_days=30).set_index("player_id")
        assert pd.isna(out.loc[1, TRANSFER_FEE_COLUMN])


class TestSelectTarget:
    def test_market_value_is_the_default_and_keeps_every_row(self, table: pd.DataFrame) -> None:
        frame, column = select_target(table)
        assert column == TARGET_COLUMN
        assert len(frame) == len(table)

    def test_transfer_fee_keeps_only_priced_rows(
        self, table: pd.DataFrame, transfers: pd.DataFrame
    ) -> None:
        """The defining property of the mode, not a defect: a row with no
        transfer has no price, and imputing one would invent the number the
        mode exists to observe."""
        frame, column = select_target(attach_transfer_fee(table, transfers), mode="transfer_fee")
        assert column == TRANSFER_FEE_COLUMN
        assert len(frame) == 1
        assert frame[TRANSFER_FEE_COLUMN].notna().all()

    def test_selecting_the_fee_before_attaching_it_raises(self, table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="run attach_transfer_fee"):
            select_target(table, mode="transfer_fee")

    def test_an_unknown_mode_raises(self, table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown target mode"):
            select_target(table, mode="salary")

    def test_the_declared_modes_are_the_ones_accepted(self) -> None:
        assert TARGET_MODES == ("market_value", "transfer_fee")
