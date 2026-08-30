"""Alternative targets, and why the default one stays the default.

The audit's fifth limitation: Transfermarkt values are community estimates, not
prices anyone paid. ``transfers.csv`` carries actual fees, so the obvious
question is whether they make a better label.

Measured on this dataset, the answer is: they make a **different** label, worth
offering and not worth defaulting to.

    transfer rows                     175,165
    with a fee above zero              17,554   (10.0%)
    joinable to a player-season         7,326   (8.5% of the training table)
    corr(log market value, log fee)     0.863
    median fee / market value           0.93

Two problems, one of which is fatal to using it as the primary target.

**Coverage.** 8.5% against the market-value label's 83%. Ten times fewer rows,
and the model that produced this project's headline numbers would lose most of
what it learns from.

**Selection.** This is the fatal one. A fee exists only where a transfer
happened, and transfers are not random: players move when a club wants to sell,
when a contract is running down, when form has spiked. Training on fees alone
learns "what does a player who was sold cost", which is not "what is this
player worth" — most players in any season are not sold, and the model would
never see them. The market-value label has no such filter: Transfermarkt values
everyone.

A fee also answers a question the market value does not. It includes contract
length, negotiating position, add-ons and a selling club's desperation. Those
are real and interesting, and none of them is in this feature set.

So it ships as an optional mode — ``--target transfer_fee`` — for anyone who
wants to model the price rather than the appraisal, with the coverage and the
bias stated where the mode is selected rather than in a footnote.
"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.build import AS_OF_COLUMN, LABEL_TIME_COLUMN, TARGET_COLUMN
from src.utils.logging import get_logger

logger = get_logger(__name__)

TRANSFER_FEE_COLUMN = "transfer_fee_eur"

TARGET_MODES = ("market_value", "transfer_fee")
"""Selectable labels. ``market_value`` is the default and the one every
published metric refers to."""


def attach_transfer_fee(
    table: pd.DataFrame, transfers: pd.DataFrame, *, tolerance_days: int = 365
) -> pd.DataFrame:
    """Attach the first paid transfer following each player-season.

    Same forward as-of discipline as the market-value label: the fee must be
    agreed *after* the season's evidence is complete, so it cannot describe a
    move that had already happened when the features were observed. Free
    transfers and loans are excluded — a zero fee is the absence of a price,
    not a price of zero, and averaging the two would teach the model that
    out-of-contract players are worthless.
    """
    fees = transfers.loc[:, ["player_id", "transfer_date", "transfer_fee"]].copy()
    fees["transfer_fee"] = pd.to_numeric(fees["transfer_fee"], errors="coerce")
    fees[LABEL_TIME_COLUMN] = pd.to_datetime(fees["transfer_date"], errors="coerce")
    fees = (
        fees[fees["transfer_fee"] > 0]
        .dropna(subset=[LABEL_TIME_COLUMN])
        .sort_values(LABEL_TIME_COLUMN)
        .rename(columns={"transfer_fee": TRANSFER_FEE_COLUMN})
        .loc[:, ["player_id", LABEL_TIME_COLUMN, TRANSFER_FEE_COLUMN]]
    )

    merged = pd.merge_asof(
        table.sort_values(AS_OF_COLUMN),
        fees,
        left_on=AS_OF_COLUMN,
        right_on=LABEL_TIME_COLUMN,
        by="player_id",
        direction="forward",
        tolerance=pd.Timedelta(tolerance_days, "D"),
        suffixes=("", "_fee"),
    )
    attached = int(merged[TRANSFER_FEE_COLUMN].notna().sum())
    logger.info(
        "transfer fee attached to %d of %d player-seasons (%.1f%%)",
        attached,
        len(merged),
        100.0 * attached / max(len(merged), 1),
    )
    return merged.reset_index(drop=True)


def select_target(table: pd.DataFrame, *, mode: str = "market_value") -> tuple[pd.DataFrame, str]:
    """The frame and target column for one target mode.

    ``transfer_fee`` drops every row without a fee, which is most of them. That
    is the mode's defining property rather than a defect: a row with no transfer
    has no price, and imputing one would invent the very number the mode exists
    to observe.

    Raises:
        ValueError: on an unknown mode, or if the fee column was never attached.
    """
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode {mode!r}; expected one of {TARGET_MODES}")
    if mode == "market_value":
        return table, TARGET_COLUMN

    if TRANSFER_FEE_COLUMN not in table.columns:
        raise ValueError(
            f"{TRANSFER_FEE_COLUMN} is absent; run attach_transfer_fee before selecting this mode"
        )
    priced = table.dropna(subset=[TRANSFER_FEE_COLUMN]).reset_index(drop=True)
    logger.warning(
        "transfer_fee mode: %d of %d rows (%.1f%%). Only players who were actually "
        "sold appear, so this answers 'what did a sold player cost', not 'what is "
        "this player worth'.",
        len(priced),
        len(table),
        100.0 * len(priced) / max(len(table), 1),
    )
    return priced, TRANSFER_FEE_COLUMN


__all__ = ["TARGET_MODES", "TRANSFER_FEE_COLUMN", "attach_transfer_fee", "select_target"]
