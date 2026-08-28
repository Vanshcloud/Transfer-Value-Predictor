"""The training table, measured against the full dataset.

Marked ``integration`` because it needs ``data/processed/`` populated by
``scripts/fetch_data.py``. It skips rather than fails when the data is absent,
so CI on a clean checkout stays green.

The figures are the feasibility spike's, reproduced here as a regression fence.
The upstream Kaggle dataset refreshes weekly, so they are asserted with a
tolerance wide enough for drift and narrow enough that a broken join — a
backward merge, a lost season boundary, a duplicated key — moves the count far
past it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.build import (
    FEATURE_COLUMNS,
    FEATURE_TIME_COLUMN,
    LABEL_TIME_COLUMN,
    TARGET_COLUMN,
    build_training_table,
    null_rates,
    select_variant,
)
from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.paths import PROJECT_ROOT
from src.validation.leakage import detect_leakage

pytestmark = pytest.mark.integration

# Spike figures (plans/01-feasibility-spike.md), ±5% for weekly refresh drift.
EXPECTED_ROWS = 37_025
EXPECTED_PLAYERS = 16_995
EXPECTED_ROWS_WITH_PRIOR = 20_030
DRIFT = 0.05


@pytest.fixture(scope="module")
def full_table() -> pd.DataFrame:
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    missing = [t for t in ("players", "player_valuations", "appearances") if not store.has_table(t)]
    if missing:
        pytest.skip(f"run scripts/fetch_data.py first; missing {missing}")

    return build_training_table(
        store.read_table("players"),
        store.read_table("player_valuations"),
        store.read_table("appearances"),
    )


def within_drift(actual: int, expected: int) -> bool:
    return abs(actual - expected) <= expected * DRIFT


def test_row_and_player_counts_match_the_spike(full_table: pd.DataFrame) -> None:
    assert within_drift(len(full_table), EXPECTED_ROWS), len(full_table)
    assert within_drift(full_table["player_id"].nunique(), EXPECTED_PLAYERS)


def test_seasons_span_2011_to_2024(full_table: pd.DataFrame) -> None:
    assert full_table["season"].min() == 2011
    assert full_table["season"].max() >= 2024


def test_the_prior_value_variant_covers_about_half_the_rows(full_table: pd.DataFrame) -> None:
    with_prior, _ = select_variant(full_table, include_prior_value=True)
    assert within_drift(len(with_prior), EXPECTED_ROWS_WITH_PRIOR), len(with_prior)


def test_null_rates_match_the_discovery_profile(full_table: pd.DataFrame) -> None:
    rates = null_rates(full_table)
    assert rates["age"] == 0.0
    assert rates["position"] == 0.0
    assert rates["minutes_played"] == 0.0
    assert rates["height_in_cm"] == pytest.approx(0.012, abs=0.005)


def test_no_row_was_built_from_data_that_postdates_its_label(full_table: pd.DataFrame) -> None:
    """The verification the whole phase exists to satisfy, on all 37,000 rows."""
    assert (full_table[FEATURE_TIME_COLUMN] <= full_table[LABEL_TIME_COLUMN]).all()


def test_one_row_per_player_season(full_table: pd.DataFrame) -> None:
    assert not full_table.duplicated(subset=["player_id", "season"]).any()


def test_the_target_is_positive_everywhere(full_table: pd.DataFrame) -> None:
    assert full_table[TARGET_COLUMN].notna().all()
    assert (full_table[TARGET_COLUMN] > 0).all()


def test_the_leakage_stage_passes_on_the_full_table(full_table: pd.DataFrame) -> None:
    for include_prior in (False, True):
        rows, columns = select_variant(full_table, include_prior_value=include_prior)
        report = detect_leakage(
            rows,
            feature_columns=columns,
            target_column=TARGET_COLUMN,
            feature_time_column=FEATURE_TIME_COLUMN,
            label_time_column=LABEL_TIME_COLUMN,
        )
        assert report.ok, report.render()


def test_the_log_transform_tames_the_target_skew(full_table: pd.DataFrame) -> None:
    """Discovery §2 measured skew 8.70 -> 0.43. Phase 6 trains on log1p."""
    import numpy as np

    raw = full_table[TARGET_COLUMN].skew()
    logged = pd.Series(np.log1p(full_table[TARGET_COLUMN])).skew()
    assert raw > 5
    assert abs(logged) < 1


def test_every_feature_column_is_present(full_table: pd.DataFrame) -> None:
    assert set(FEATURE_COLUMNS) <= set(full_table.columns)


def test_career_features_do_not_extrapolate_past_the_training_range(
    full_table: pd.DataFrame,
) -> None:
    """The reason `years_since_debut` and `seasons_observed` are capped.

    Coverage starts in 2012, so both grow in lockstep with the calendar until
    the cap binds. Without it a 2024 test row carries a value no row in a
    <=2021 training set could ever hold, and the model extrapolates off the end
    of its training range — the exact failure that excluding raw `season`
    was meant to prevent.

    `seasons_observed` retains a small residual because a count advances at a
    player-specific rate, so a fixed ceiling cannot equalise its support
    exactly. The fence keeps that residual negligible rather than pretending
    it is zero.
    """
    train = full_table[full_table["season"] <= 2021]
    test = full_table[full_table["season"] >= 2023]

    beyond = (test["years_since_debut"] > train["years_since_debut"].max()).mean()
    assert beyond == 0.0

    beyond = (test["seasons_observed"] > train["seasons_observed"].max()).mean()
    assert beyond < 0.005, f"{beyond:.2%} of test rows exceed the training range"
