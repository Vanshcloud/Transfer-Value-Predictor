"""Error analysis: where the model is wrong, and on whom.

One MAE says the model is off by EUR 3.7M. It does not say that the error sits
almost entirely in the top value band, which is the thing that decides whether
a prediction can be trusted for the player actually being asked about.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    AGE_BANDS,
    VALUE_BANDS,
    analyse_errors,
    build_residuals,
    segment_errors,
)
from src.feature_engineering.build import TARGET_COLUMN


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    size = 400
    return pd.DataFrame(
        {
            "player_id": range(size),
            "season": rng.choice([2023, 2024], size),
            "position": rng.choice(["Attack", "Defender", "Goalkeeper"], size),
            "age": rng.uniform(18, 36, size),
            TARGET_COLUMN: rng.uniform(1e5, 6e7, size),
        }
    )


class TestResiduals:
    def test_residual_is_predicted_minus_actual(self) -> None:
        """Stated because the opposite convention is equally common."""
        frame = pd.DataFrame({TARGET_COLUMN: [1_000_000.0], "age": [25.0]})
        residuals = build_residuals(frame, np.array([1_500_000.0]), target_column=TARGET_COLUMN)
        assert residuals["residual"].iloc[0] == 500_000
        assert residuals["absolute_error"].iloc[0] == 500_000

    def test_a_positive_residual_means_the_model_overvalued(self) -> None:
        frame = pd.DataFrame({TARGET_COLUMN: [2_000_000.0], "age": [25.0]})
        residuals = build_residuals(frame, np.array([500_000.0]), target_column=TARGET_COLUMN)
        assert residuals["residual"].iloc[0] < 0  # undervalued

    def test_every_row_lands_in_a_value_band(self, frame: pd.DataFrame) -> None:
        residuals = build_residuals(
            frame, frame[TARGET_COLUMN].to_numpy(), target_column=TARGET_COLUMN
        )
        assert residuals["value_band"].notna().all()
        assert set(residuals["value_band"]) <= {name for name, _, _ in VALUE_BANDS}

    def test_every_row_lands_in_an_age_band(self, frame: pd.DataFrame) -> None:
        residuals = build_residuals(
            frame, frame[TARGET_COLUMN].to_numpy(), target_column=TARGET_COLUMN
        )
        assert residuals["age_band"].notna().all()
        assert set(residuals["age_band"]) <= {name for name, _, _ in AGE_BANDS}

    def test_bands_are_left_closed_and_right_open(self) -> None:
        frame = pd.DataFrame({TARGET_COLUMN: [1e6], "age": [21.0]})
        residuals = build_residuals(frame, np.array([1e6]), target_column=TARGET_COLUMN)
        assert residuals["value_band"].iloc[0] == "1M-5M"
        assert residuals["age_band"].iloc[0] == "21-24"

    def test_a_frame_without_age_still_works(self) -> None:
        frame = pd.DataFrame({TARGET_COLUMN: [1e6]})
        residuals = build_residuals(frame, np.array([1e6]), target_column=TARGET_COLUMN)
        assert residuals["age_band"].isna().all()

    def test_percentage_error_does_not_divide_by_zero(self) -> None:
        frame = pd.DataFrame({TARGET_COLUMN: [0.0], "age": [25.0]})
        residuals = build_residuals(frame, np.array([1e6]), target_column=TARGET_COLUMN)
        assert pd.isna(residuals["percentage_error"].iloc[0])


class TestSegments:
    def test_thin_slices_are_dropped(self, frame: pd.DataFrame) -> None:
        """An MAE over eleven players is a number people would quote."""
        residuals = build_residuals(
            frame, frame[TARGET_COLUMN].to_numpy(), target_column=TARGET_COLUMN
        )
        segments = segment_errors(residuals, target_column=TARGET_COLUMN, min_rows=1000)
        assert segments == []

    def test_it_reports_each_requested_segment(self, frame: pd.DataFrame) -> None:
        residuals = build_residuals(
            frame, frame[TARGET_COLUMN].to_numpy(), target_column=TARGET_COLUMN
        )
        # min_rows=2, not 1: R^2 over a single row is undefined and sklearn
        # rightly warns. The product default is 30.
        segments = segment_errors(residuals, target_column=TARGET_COLUMN, min_rows=2)
        assert {s.segment for s in segments} == {"value_band", "age_band", "position", "season"}

    def test_a_missing_segment_column_is_skipped_not_fatal(self, frame: pd.DataFrame) -> None:
        residuals = build_residuals(
            frame, frame[TARGET_COLUMN].to_numpy(), target_column=TARGET_COLUMN
        )
        segments = segment_errors(
            residuals, target_column=TARGET_COLUMN, by=("nonexistent",), min_rows=1
        )
        assert segments == []


class TestAnalyseErrors:
    def test_a_perfect_model_has_no_error(self, frame: pd.DataFrame) -> None:
        analysis = analyse_errors(
            frame, frame[TARGET_COLUMN].to_numpy(dtype=float), target_column=TARGET_COLUMN
        )
        assert analysis.overall.mae == 0.0
        assert all(s.mae == 0.0 for s in analysis.segments)

    def test_the_two_miss_lists_are_opposite_failure_modes(self, frame: pd.DataFrame) -> None:
        # Sorted by magnitude alone, both lists would be the same rows.
        rng = np.random.default_rng(1)
        predictions = frame[TARGET_COLUMN].to_numpy(dtype=float) + rng.normal(0, 5e6, len(frame))
        analysis = analyse_errors(frame, predictions, target_column=TARGET_COLUMN)

        assert (analysis.worst_overpredictions["residual"] > 0).all()
        assert (analysis.worst_underpredictions["residual"] < 0).all()

    def test_overpredictions_are_ordered_worst_first(self, frame: pd.DataFrame) -> None:
        rng = np.random.default_rng(1)
        predictions = frame[TARGET_COLUMN].to_numpy(dtype=float) + rng.normal(0, 5e6, len(frame))
        analysis = analyse_errors(frame, predictions, target_column=TARGET_COLUMN)

        residuals = list(analysis.worst_overpredictions["residual"])
        assert residuals == sorted(residuals, reverse=True)

    def test_it_serialises_to_json(self, frame: pd.DataFrame) -> None:
        analysis = analyse_errors(
            frame, frame[TARGET_COLUMN].to_numpy(dtype=float), target_column=TARGET_COLUMN
        )
        restored = json.loads(json.dumps(analysis.as_dict()))
        assert restored["overall"]["n"] == len(frame)
        assert restored["segments"]

    def test_segments_for_filters_by_name(self, frame: pd.DataFrame) -> None:
        analysis = analyse_errors(
            frame, frame[TARGET_COLUMN].to_numpy(dtype=float), target_column=TARGET_COLUMN
        )
        assert all(s.segment == "position" for s in analysis.segments_for("position"))
