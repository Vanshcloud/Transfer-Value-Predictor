"""Validation checks and the per-table rule sets.

Two tests here exist because of specific traps measured in the real data, and
are marked as such: the literal "Missing" position string, and the composite
primary key on player_valuations.
"""

from __future__ import annotations

import pandas as pd
import pytest

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
from src.validation.report import Finding, Severity, ValidationError, ValidationReport
from src.validation.tables import (
    KNOWN_POSITIONS,
    validate_appearances,
    validate_player_valuations,
    validate_players,
)

# --- report primitives ---


def test_empty_report_is_ok() -> None:
    report = ValidationReport()
    assert report.ok
    report.raise_for_errors()


def test_warnings_do_not_make_a_report_fail() -> None:
    report = ValidationReport([Finding("c", Severity.WARNING, "t", "noticed")])
    assert report.ok
    report.raise_for_errors()


def test_errors_raise_and_list_every_error() -> None:
    report = ValidationReport(
        [
            Finding("a", Severity.ERROR, "t", "first"),
            Finding("b", Severity.ERROR, "t", "second"),
        ]
    )
    assert not report.ok
    with pytest.raises(ValidationError, match="2 validation error"):
        report.raise_for_errors()


def test_count_unit_is_honoured() -> None:
    """A finding about columns must not report itself as rows."""
    finding = Finding("c", Severity.ERROR, "t", "bad", count=3, unit="columns")
    assert "3 columns" in finding.render()


def test_numpy_scalars_render_plainly() -> None:
    import numpy as np

    finding = Finding("c", Severity.WARNING, "t", "m", 1, (np.float64(17.0),))
    assert "np.float64" not in finding.render()


# --- individual checks ---


def test_required_columns_reports_what_is_absent() -> None:
    frame = pd.DataFrame({"a": [1]})
    findings = check_required_columns(frame, "t", ["a", "b", "c"])
    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert set(findings[0].examples) == {"b", "c"}


def test_not_empty() -> None:
    assert check_not_empty(pd.DataFrame({"a": [1]}), "t") == []
    assert len(check_not_empty(pd.DataFrame({"a": []}), "t")) == 1


def test_primary_key_accepts_a_unique_key() -> None:
    frame = pd.DataFrame({"id": [1, 2, 3]})
    assert check_primary_key(frame, "t", ["id"]) == []


def test_primary_key_reports_duplicates() -> None:
    frame = pd.DataFrame({"id": [1, 1, 2]})
    findings = check_primary_key(frame, "t", ["id"])
    assert findings[0].severity is Severity.ERROR
    assert findings[0].count == 2  # both copies are reported


def test_composite_primary_key_distinguishes_the_pair_from_the_parts() -> None:
    """THE TRAP: player_valuations is keyed on (player_id, date), not player_id.

    Measured on the real table: 614,773 rows duplicate on player_id alone and
    zero duplicate on the pair. A player is valued many times — median 15,
    maximum 57 — because a row is written when the value *changes*. Validating
    against player_id alone would report a catastrophic failure on correct data.
    """
    frame = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "date": ["2020-01-01", "2020-06-01", "2021-01-01", "2020-01-01"],
        }
    )
    assert check_primary_key(frame, "t", ["player_id"]), "player_id alone should look duplicated"
    assert check_primary_key(frame, "t", ["player_id", "date"]) == [], "the pair is unique"


def test_no_nulls() -> None:
    frame = pd.DataFrame({"a": [1, None, 3]})
    assert check_no_nulls(frame, "t", ["a"])[0].count == 1
    assert check_no_nulls(pd.DataFrame({"a": [1]}), "t", ["a"]) == []


def test_sentinel_strings_catch_what_isna_misses() -> None:
    """THE TRAP: `position` uses the literal string "Missing", never NaN.

    586 rows in the real table. A null check written with isna() alone passes
    straight over them, and the model then learns "Missing" as a real position.
    """
    frame = pd.DataFrame({"position": ["Attack", "Missing", "Defender"]})
    assert frame["position"].isna().sum() == 0, "precondition: nothing is NaN"

    assert check_no_nulls(frame, "players", ["position"]) == [], "isna() sees nothing"

    findings = check_sentinel_strings(frame, "players", ["position"])
    assert len(findings) == 1
    assert findings[0].count == 1
    assert "Missing" in findings[0].examples


def test_sentinel_check_ignores_numeric_columns() -> None:
    frame = pd.DataFrame({"n": [1, 2, 3]})
    assert check_sentinel_strings(frame, "t", ["n"]) == []


def test_allowed_values_flags_new_levels() -> None:
    frame = pd.DataFrame({"position": ["Attack", "Sweeper"]})
    findings = check_allowed_values(frame, "t", "position", KNOWN_POSITIONS)
    assert findings[0].examples == ("Sweeper",)


def test_allowed_values_ignores_nulls() -> None:
    frame = pd.DataFrame({"position": ["Attack", None]})
    assert check_allowed_values(frame, "t", "position", KNOWN_POSITIONS) == []


@pytest.mark.parametrize(
    ("values", "minimum", "maximum", "expected"),
    [
        ([150, 180, 200], 140, 220, 0),
        ([17, 180], 140, 220, 1),
        ([250, 180], 140, 220, 1),
        ([17, 250], 140, 220, 2),
        ([-1, 0, 1], 0, None, 1),
    ],
)
def test_range(values: list[int], minimum: int, maximum: int | None, expected: int) -> None:
    frame = pd.DataFrame({"v": values})
    findings = check_range(frame, "t", "v", minimum=minimum, maximum=maximum)
    assert (findings[0].count if findings else 0) == expected


def test_range_ignores_nulls() -> None:
    frame = pd.DataFrame({"v": [None, 180.0]})
    assert check_range(frame, "t", "v", minimum=140, maximum=220) == []


def test_parseable_dates() -> None:
    frame = pd.DataFrame({"d": ["2020-01-01", "not-a-date", None]})
    findings = check_parseable_dates(frame, "t", ["d"])
    assert findings[0].count == 1, "the pre-existing null is not a parse failure"


# --- table rule sets, exercised against the committed sample ---


def test_sample_players_validate(sample_players: pd.DataFrame) -> None:
    report = validate_players(sample_players)
    assert report.ok, report.render()


def test_sample_valuations_validate(sample_valuations: pd.DataFrame) -> None:
    report = validate_player_valuations(sample_valuations)
    assert report.ok, report.render()


def test_sample_appearances_validate(sample_appearances: pd.DataFrame) -> None:
    report = validate_appearances(sample_appearances)
    assert report.ok, report.render()


def test_valuations_reject_a_non_positive_value(sample_valuations: pd.DataFrame) -> None:
    """A value <= 0 breaks the log1p target transform, so it is an error."""
    broken = sample_valuations.copy()
    broken.loc[broken.index[0], "market_value_in_eur"] = 0
    report = validate_player_valuations(broken)
    assert not report.ok


def test_valuations_reject_a_duplicate_pair(sample_valuations: pd.DataFrame) -> None:
    broken = pd.concat([sample_valuations, sample_valuations.head(1)], ignore_index=True)
    assert not validate_player_valuations(broken).ok


def test_appearances_reject_negative_minutes(sample_appearances: pd.DataFrame) -> None:
    broken = sample_appearances.copy()
    broken.loc[broken.index[0], "minutes_played"] = -5
    assert not validate_appearances(broken).ok


def test_players_missing_a_required_column_is_an_error(sample_players: pd.DataFrame) -> None:
    assert not validate_players(sample_players.drop(columns=["player_id"])).ok


# --- the pipeline stage ---


def test_validate_stage_passes_on_the_sample(tmp_path: object) -> None:
    """The stage runs green on data that satisfies its contract."""
    from pathlib import Path

    from src.pipelines.validate import validate
    from src.storage.duckdb_store import DuckDBParquetStore
    from src.utils.paths import PROJECT_ROOT

    store = DuckDBParquetStore(Path(str(tmp_path)) / "processed")
    for name in ("players", "player_valuations", "appearances"):
        store.write_table(name, pd.read_csv(PROJECT_ROOT / "data" / "sample" / f"{name}.csv"))

    report = validate(store)
    assert report.ok, report.render()


def test_validate_stage_raises_on_a_contract_violation(tmp_path: object) -> None:
    from pathlib import Path

    from src.pipelines.validate import validate
    from src.storage.duckdb_store import DuckDBParquetStore
    from src.utils.paths import PROJECT_ROOT

    store = DuckDBParquetStore(Path(str(tmp_path)) / "processed")
    valuations = pd.read_csv(PROJECT_ROOT / "data" / "sample" / "player_valuations.csv")
    valuations.loc[valuations.index[0], "market_value_in_eur"] = -1
    store.write_table("player_valuations", valuations)

    with pytest.raises(ValidationError):
        validate(store)


def test_validate_stage_skips_absent_tables(tmp_path: object) -> None:
    """A partially populated store must not crash the stage."""
    from pathlib import Path

    from src.pipelines.validate import validate
    from src.storage.duckdb_store import DuckDBParquetStore
    from src.utils.paths import PROJECT_ROOT

    store = DuckDBParquetStore(Path(str(tmp_path)) / "processed")
    store.write_table("players", pd.read_csv(PROJECT_ROOT / "data" / "sample" / "players.csv"))
    assert validate(store).ok
