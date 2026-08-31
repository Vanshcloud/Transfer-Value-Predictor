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
    build_current_season_table,
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

CURRENT_SEASON_TABLE = "current_season"
"""Feature rows for seasons too recent to have a label.

Written beside the training table and never mixed into it. The service reads it
so a prediction can be asked about the season being played rather than only
about the last one Transfermarkt has finished revaluing.
"""

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
"""The three tables the training table cannot be built without."""

CONTEXT_TABLES = ("competitions", "games", "club_games", "game_lineups")
"""Optional enrichment: competition identity, club results, squad role.

Optional because the committed sample data carries only the three required
tables, and the suite has to build a valid table from it. A build without these
produces the same rows with the context and role features null — which the
fitted imputer handles — rather than failing.
"""


@dataclass(frozen=True)
class FeatureReport:
    """What one feature-engineering run produced."""

    rows: int
    players: int
    first_season: int
    last_season: int
    rows_with_prior_value: int
    current_season_rows: int
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
                f"  current-season rows      {self.current_season_rows:>10,}  (unlabelled, "
                f"predictable)",
                "  null feature rates (non-zero only):",
                nulls or "    none",
                f"  leakage: {len(self.leakage.errors)} error(s), "
                f"{len(self.leakage.warnings)} warning(s)",
            ]
        )


def build_features(
    store: TableStore, *, season_start_month: int = 8, tolerance_days: int = 365
) -> FeatureReport:
    """Build the training table from ``store`` and write it back to ``store``.

    Raises:
        KeyError: if ingestion has not run.
        ValidationError: if the leakage stage finds a violation. The table is
            not written in that case — a leaky table on disk is worse than no
            table, because the next stage cannot tell the difference.
    """
    frames = {name: store.read_table(name) for name in SOURCE_TABLES}

    # Enrichment tables are read when present and skipped when not, so the
    # same code path serves the full download and the committed sample.
    context: dict[str, pd.DataFrame] = {}
    for name in CONTEXT_TABLES:
        if store.has_table(name):
            context[name] = store.read_table(name)
        else:
            logger.warning("no %s table; its features will be null", name)

    table = build_training_table(
        frames["players"],
        frames["player_valuations"],
        frames["appearances"],
        competitions=context.get("competitions"),
        games=context.get("games"),
        club_games=context.get("club_games"),
        lineups=context.get("game_lineups"),
        season_start_month=season_start_month,
        label_tolerance_days=tolerance_days,
    )

    # Every row, not only the prior-value variant's.
    #
    # This used to validate `select_variant(include_prior_value=True)` on the
    # grounds that it "carries every column the performance-only variant does,
    # plus the lagged target most likely to be mis-specified". True of the
    # columns and false of the rows: that frame is 61,522 of 85,966, so the
    # 24,444 rows with no prior value — 28.4%, and every player's first season
    # among them — were never checked for duplicate entities, feature/label
    # ordering or future-dated values. They pass today. Nothing was making them.
    with_prior, prior_columns = select_variant(table, include_prior_value=True)
    _, all_columns = select_variant(table, include_prior_value=False)

    leakage = leakage_validator(all_columns).validate(table)
    leakage.extend(leakage_validator(prior_columns).validate(with_prior).findings)
    for finding in leakage.warnings:
        logger.warning("%s", finding.render())
    leakage.raise_for_errors()

    store.write_table(TRAINING_TABLE, table)

    # Rows for the season(s) after the last labelled one. They cannot be
    # trained on — they have no target — but every feature they need is already
    # known, so there is no reason the service cannot price them.
    current = build_current_season_table(
        frames["players"],
        frames["player_valuations"],
        frames["appearances"],
        table,
        competitions=context.get("competitions"),
        games=context.get("games"),
        club_games=context.get("club_games"),
        lineups=context.get("game_lineups"),
        season_start_month=season_start_month,
    )
    if not current.empty:
        # The serving table gets the same contract as the training table.
        # It had none: these rows are what `/api/v1/predict` answers from, and
        # nothing checked that a feature on them predates the moment they claim
        # to describe. They have no label, so the label-ordering checks skip
        # themselves; the duplicate, current-state and lagged-value checks all
        # still apply and are the ones that would catch a bad join here.
        current_leakage = leakage_validator(all_columns).validate(current)
        for finding in current_leakage.warnings:
            logger.warning("current_season: %s", finding.render())
        current_leakage.raise_for_errors()
        store.write_table(CURRENT_SEASON_TABLE, current)

    return FeatureReport(
        rows=len(table),
        players=int(table["player_id"].nunique()),
        first_season=int(table["season"].min()),
        last_season=int(table["season"].max()),
        rows_with_prior_value=len(with_prior),
        current_season_rows=len(current),
        null_rates=null_rates(table),
        leakage=leakage,
    )
