"""Per-table rule sets for the Kaggle source.

Every bound here was measured against the real data rather than assumed; the
figures in the comments are from that profiling run and are what the tests
assert against.
"""

from __future__ import annotations

import pandas as pd

from src.storage.base import TableStore
from src.utils.logging import get_logger
from src.validation.checks import (
    check_allowed_values,
    check_no_nulls,
    check_not_empty,
    check_parseable_dates,
    check_primary_key,
    check_range,
    check_required_columns,
    check_sentinel_strings,
)
from src.validation.report import Severity, ValidationReport

logger = get_logger(__name__)

# The five values `position` actually takes. "Missing" is one of them — a
# literal string, not NaN — which is why it is listed as known rather than
# flagged as an unexpected level. check_sentinel_strings is what reports it.
KNOWN_POSITIONS = frozenset({"Attack", "Defender", "Goalkeeper", "Midfield", "Missing"})
KNOWN_FEET = frozenset({"left", "right", "both"})

# Plausible senior-professional heights. Measured: 13 players carry 17, 18 or
# 19 cm — digits dropped from 170/180/190 — against a p0.1 of 160 and a max of
# 210. A warning, not an error: 0.026% of rows, and preprocessing nulls them.
MIN_HEIGHT_CM, MAX_HEIGHT_CM = 140, 220


def validate_players(frame: pd.DataFrame) -> ValidationReport:
    """players.csv — one row per player, keyed on player_id."""
    report = ValidationReport()
    table = "players"

    report.extend(
        check_required_columns(
            frame,
            table,
            ["player_id", "name", "date_of_birth", "position", "country_of_citizenship"],
        )
    )
    report.extend(check_not_empty(frame, table))
    report.extend(check_primary_key(frame, table, ["player_id"]))
    report.extend(check_no_nulls(frame, table, ["player_id", "name"]))
    # 49 players lack a date of birth (0.1%). Age is a feature, so those rows
    # cannot be modelled — but the table itself is not broken.
    report.extend(check_no_nulls(frame, table, ["date_of_birth"], severity=Severity.WARNING))
    report.extend(
        check_parseable_dates(frame, table, ["date_of_birth", "contract_expiration_date"])
    )
    report.extend(check_sentinel_strings(frame, table, ["position", "sub_position", "foot"]))
    report.extend(check_allowed_values(frame, table, "position", KNOWN_POSITIONS))
    report.extend(check_allowed_values(frame, table, "foot", KNOWN_FEET))
    report.extend(
        check_range(frame, table, "height_in_cm", minimum=MIN_HEIGHT_CM, maximum=MAX_HEIGHT_CM)
    )
    report.extend(check_range(frame, table, "international_caps", minimum=0))
    report.extend(check_range(frame, table, "international_goals", minimum=0))
    return report


def validate_player_valuations(frame: pd.DataFrame) -> ValidationReport:
    """player_valuations.csv — the label.

    The primary key is (player_id, date), NOT player_id. Measured: 614,773 rows
    duplicate on player_id alone and zero duplicate on the pair. A valuation is
    recorded when a player's value *changes*, so a player has many — a median of
    15 and a maximum of 57. Treating player_id as the key here would look like a
    catastrophic duplicate-key failure on data that is entirely correct.
    """
    report = ValidationReport()
    table = "player_valuations"

    report.extend(
        check_required_columns(frame, table, ["player_id", "date", "market_value_in_eur"])
    )
    report.extend(check_not_empty(frame, table))
    report.extend(check_primary_key(frame, table, ["player_id", "date"]))
    report.extend(check_no_nulls(frame, table, ["player_id", "date", "market_value_in_eur"]))
    report.extend(check_parseable_dates(frame, table, ["date"]))
    # Measured minimum is 10,000 EUR with no nulls and nothing <= 0. A
    # non-positive market value would break the log1p target transform.
    report.extend(
        check_range(frame, table, "market_value_in_eur", minimum=1, severity=Severity.ERROR)
    )
    return report


def validate_appearances(frame: pd.DataFrame) -> ValidationReport:
    """appearances.csv — per-player-per-match performance, the feature source."""
    report = ValidationReport()
    table = "appearances"

    report.extend(
        check_required_columns(
            frame,
            table,
            ["appearance_id", "player_id", "date", "minutes_played", "goals", "assists"],
        )
    )
    report.extend(check_not_empty(frame, table))
    report.extend(check_primary_key(frame, table, ["appearance_id"]))
    report.extend(check_no_nulls(frame, table, ["player_id", "date"]))
    report.extend(check_parseable_dates(frame, table, ["date"]))
    for column in ("goals", "assists", "minutes_played", "yellow_cards", "red_cards"):
        report.extend(check_range(frame, table, column, minimum=0, severity=Severity.ERROR))
    # A match is 90 minutes plus stoppage and extra time; the measured maximum
    # is 148. Anything beyond 150 is a data error, not a long match.
    report.extend(check_range(frame, table, "minutes_played", maximum=150))
    return report


VALIDATORS = {
    "players": validate_players,
    "player_valuations": validate_player_valuations,
    "appearances": validate_appearances,
}


def validate_store(store: TableStore) -> ValidationReport:
    """Validate every known table present in ``store``."""
    report = ValidationReport()
    for name, validator in VALIDATORS.items():
        if not store.has_table(name):
            logger.warning("validation: table %s is absent from the store", name)
            continue
        report.extend(validator(store.read_table(name)).findings)
    return report
