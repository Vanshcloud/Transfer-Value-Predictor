"""Leakage detection.

Leakage is the failure mode that does not announce itself: it makes a metric
better, and nobody investigates a model that beat expectations. These tests
assert each detector fires on a leak and stays quiet on a correctly built
table, because a detector that cries wolf gets switched off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.leakage import (
    CURRENT_STATE_COLUMNS,
    check_feature_time_precedes_label,
    check_no_current_state_columns,
    check_splits_are_disjoint,
    check_target_absent_from_features,
    detect_leakage,
)
from src.validation.report import Severity


@pytest.fixture
def clean_table() -> pd.DataFrame:
    """A correctly built table: features observed before the label is set."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "season_end": pd.to_datetime(["2021-07-01"] * 4),
            "label_date": pd.to_datetime(["2021-08-15"] * 4),
            "goals": [10, 5, 0, 2],
            "prev_market_value_in_eur": [1e6, 2e6, 3e6, 4e6],
            "market_value_in_eur": [2e6, 3e6, 4e6, 5e6],
        }
    )


# --- 1. feature observed after its label ---


def test_clean_ordering_passes(clean_table: pd.DataFrame) -> None:
    assert (
        check_feature_time_precedes_label(
            clean_table, feature_time_column="season_end", label_time_column="label_date"
        )
        == []
    )


def test_feature_after_label_is_an_error(clean_table: pd.DataFrame) -> None:
    leaky = clean_table.copy()
    leaky["season_end"] = leaky["label_date"] + pd.Timedelta(30, "D")
    findings = check_feature_time_precedes_label(
        leaky, feature_time_column="season_end", label_time_column="label_date"
    )
    assert findings[0].severity is Severity.ERROR
    assert findings[0].count == 4


def test_equal_timestamps_are_allowed(clean_table: pd.DataFrame) -> None:
    """Observed at the same instant is not observed afterwards."""
    same = clean_table.copy()
    same["season_end"] = same["label_date"]
    assert (
        check_feature_time_precedes_label(
            same, feature_time_column="season_end", label_time_column="label_date"
        )
        == []
    )


def test_missing_time_column_is_reported_not_ignored(clean_table: pd.DataFrame) -> None:
    """Silently skipping the check would be the worst outcome."""
    findings = check_feature_time_precedes_label(
        clean_table, feature_time_column="absent", label_time_column="label_date"
    )
    assert findings[0].severity is Severity.ERROR


# --- 2. current-state columns on historical rows ---


def test_contract_expiration_date_is_rejected() -> None:
    """The specific offender: players.csv holds the CURRENT contract, so on a
    2015 row it tells the model something nobody knew in 2015."""
    frame = pd.DataFrame({"goals": [1], "contract_expiration_date": ["2027-06-30"]})
    findings = check_no_current_state_columns(frame)
    assert findings[0].severity is Severity.ERROR
    assert "contract_expiration_date" in findings[0].examples
    assert findings[0].unit == "columns"


def test_clean_features_pass(clean_table: pd.DataFrame) -> None:
    assert check_no_current_state_columns(clean_table[["goals", "player_id"]]) == []


@pytest.mark.parametrize("column", sorted(CURRENT_STATE_COLUMNS))
def test_every_declared_current_state_column_is_caught(column: str) -> None:
    assert check_no_current_state_columns(pd.DataFrame({column: [1]}))


# --- 3. target reaching the feature matrix ---


def test_raw_target_as_a_feature_is_caught() -> None:
    findings = check_target_absent_from_features(
        ["goals", "market_value_in_eur"], "market_value_in_eur"
    )
    assert findings[0].severity is Severity.ERROR


def test_transformed_target_is_caught() -> None:
    """The realistic version: it arrives renamed, not as the raw column."""
    findings = check_target_absent_from_features(
        ["goals", "log_market_value_in_eur"], "market_value_in_eur"
    )
    assert "log_market_value_in_eur" in findings[0].examples


def test_properly_lagged_feature_is_allowed() -> None:
    """prev_market_value_in_eur is legitimate and must not be flagged, or the
    check gets disabled the first time someone adds a lag feature."""
    assert (
        check_target_absent_from_features(
            ["goals", "prev_market_value_in_eur"], "market_value_in_eur"
        )
        == []
    )


@pytest.mark.parametrize("prefix", ["prev_", "lag_", "prior_"])
def test_each_lag_prefix_is_allowed(prefix: str) -> None:
    assert (
        check_target_absent_from_features([f"{prefix}market_value_in_eur"], "market_value_in_eur")
        == []
    )


# --- 4. split overlap ---


def test_disjoint_splits_pass() -> None:
    splits = {"train": pd.Index([0, 1, 2]), "test": pd.Index([3, 4])}
    assert check_splits_are_disjoint(splits) == []


def test_overlapping_rows_are_an_error() -> None:
    splits = {"train": pd.Index([0, 1, 2]), "test": pd.Index([2, 3])}
    findings = check_splits_are_disjoint(splits)
    assert findings[0].severity is Severity.ERROR
    assert findings[0].count == 1


def test_three_way_split_checks_every_pair() -> None:
    splits = {
        "train": pd.Index([0, 1]),
        "validation": pd.Index([1, 2]),
        "test": pd.Index([2, 3]),
    }
    assert len(check_splits_are_disjoint(splits)) == 2


def test_shared_players_are_a_warning_not_an_error() -> None:
    """Under a temporal split a career legitimately spans the boundary, so this
    is information, not a failure."""
    groups = pd.Series([1, 1, 2, 2], index=[0, 1, 2, 3])
    splits = {"train": pd.Index([0, 2]), "test": pd.Index([1, 3])}
    findings = check_splits_are_disjoint(splits, groups=groups)
    assert [f.severity for f in findings] == [Severity.WARNING]
    assert findings[0].unit == "players"


# --- the composed stage ---


def test_detect_leakage_passes_a_clean_table(clean_table: pd.DataFrame) -> None:
    report = detect_leakage(
        clean_table,
        feature_columns=["goals", "prev_market_value_in_eur"],
        target_column="market_value_in_eur",
        feature_time_column="season_end",
        label_time_column="label_date",
        splits={"train": pd.Index([0, 1]), "test": pd.Index([2, 3])},
    )
    assert report.ok, report.render()


def test_detect_leakage_catches_all_four_modes(clean_table: pd.DataFrame) -> None:
    leaky = clean_table.copy()
    leaky["season_end"] = leaky["label_date"] + pd.Timedelta(30, "D")
    leaky["contract_expiration_date"] = "2027-06-30"
    leaky["log_market_value_in_eur"] = np.log1p(leaky["market_value_in_eur"])

    report = detect_leakage(
        leaky,
        feature_columns=["goals", "contract_expiration_date", "log_market_value_in_eur"],
        target_column="market_value_in_eur",
        feature_time_column="season_end",
        label_time_column="label_date",
        splits={"train": pd.Index([0, 1, 2]), "test": pd.Index([2, 3])},
    )
    assert not report.ok
    checks = {f.check for f in report.errors}
    assert checks == {
        "leakage_feature_time",
        "leakage_current_state",
        "leakage_target_in_features",
        "leakage_split_overlap",
    }
