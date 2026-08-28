"""Feature engineering as a pipeline stage.

Reads the validated raw tables, builds the training table, and refuses to write
it if the leakage stage finds anything. The check runs *here*, in the pipeline,
not only in the test suite — a leak that only a test catches is a leak that
ships the moment someone builds the table by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.feature_engineering.build import (
    FEATURE_TIME_COLUMN,
    LABEL_TIME_COLUMN,
    TARGET_COLUMN,
    build_training_table,
    null_rates,
    select_variant,
)
from src.storage.base import TableStore
from src.utils.logging import get_logger
from src.validation.leakage import LeakageValidator
from src.validation.report import ValidationReport

logger = get_logger(__name__)

TRAINING_TABLE = "training_table"

ENTITY_KEYS = ("player_id", "season")
"""What makes a row unique. Two rows for one player-season are one observation
twice, and they would straddle any split that is not grouped by player."""


def leakage_validator(feature_columns: tuple[str, ...]) -> LeakageValidator:
    """The validator this project's training table is checked against.

    Built here rather than inline so Phase 6 onward re-runs the identical
    contract after splitting, instead of a hand-copied subset of it.
    """
    return LeakageValidator(
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        feature_time_column=FEATURE_TIME_COLUMN,
        label_time_column=LABEL_TIME_COLUMN,
        entity_keys=ENTITY_KEYS,
        table=TRAINING_TABLE,
    )


SOURCE_TABLES = ("players", "player_valuations", "appearances")


@dataclass(frozen=True)
class FeatureReport:
    """What one feature-engineering run produced."""

    rows: int
    players: int
    first_season: int
    last_season: int
    rows_with_prior_value: int
    null_rates: pd.Series
    leakage: ValidationReport

    def render(self) -> str:
        nulls = "\n".join(
            f"    {name:<24} {rate:6.2%}" for name, rate in self.null_rates.items() if rate
        )
        return "\n".join(
            [
                f"  rows                     {self.rows:>10,}",
                f"  players                  {self.players:>10,}",
                f"  seasons                  {self.first_season:>10}-{self.last_season}",
                f"  rows with prior value    {self.rows_with_prior_value:>10,}",
                "  null feature rates (non-zero only):",
                nulls or "    none",
                f"  leakage: {len(self.leakage.errors)} error(s), "
                f"{len(self.leakage.warnings)} warning(s)",
            ]
        )


def build_features(
    store: TableStore, *, season_start_month: int = 8, tolerance_days: int = 120
) -> FeatureReport:
    """Build the training table from ``store`` and write it back to ``store``.

    Raises:
        KeyError: if ingestion has not run.
        ValidationError: if the leakage stage finds a violation. The table is
            not written in that case — a leaky table on disk is worse than no
            table, because the next stage cannot tell the difference.
    """
    frames = {name: store.read_table(name) for name in SOURCE_TABLES}

    table = build_training_table(
        frames["players"],
        frames["player_valuations"],
        frames["appearances"],
        season_start_month=season_start_month,
        label_tolerance_days=tolerance_days,
    )

    # Check the harder of the two variants: it carries every column the
    # performance-only variant does, plus the lagged target most likely to be
    # mis-specified.
    with_prior, feature_columns = select_variant(table, include_prior_value=True)
    leakage = leakage_validator(feature_columns).validate(with_prior)
    for finding in leakage.warnings:
        logger.warning("%s", finding.render())
    leakage.raise_for_errors()

    store.write_table(TRAINING_TABLE, table)

    return FeatureReport(
        rows=len(table),
        players=int(table["player_id"].nunique()),
        first_season=int(table["season"].min()),
        last_season=int(table["season"].max()),
        rows_with_prior_value=len(with_prior),
        null_rates=null_rates(table),
        leakage=leakage,
    )
