"""Where the model is wrong, and on whom.

A single MAE says the model is off by EUR 3.7M on average. It does not say that
the error is concentrated in the top value decile, or that goalkeepers are
systematically undervalued, and those are the things that decide whether a
prediction can be trusted for a given player. Everything here returns frames
and plain structures so the API and the dashboard can serve the same breakdown
the report renders.

Residuals are in EUR and signed **predicted minus actual**, so a positive
residual means the model paid too much. That convention is stated because the
opposite one is equally common and silently flips every conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import Metrics, evaluate

VALUE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<1M", 0, 1e6),
    ("1M-5M", 1e6, 5e6),
    ("5M-20M", 5e6, 2e7),
    ("20M-50M", 2e7, 5e7),
    (">50M", 5e7, float("inf")),
)
"""Market-value bands. Uneven on purpose: the target spans four orders of
magnitude, so equal-width bands would put 95% of players in one bucket."""

AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<21", 0, 21),
    ("21-24", 21, 25),
    ("25-28", 25, 29),
    ("29-32", 29, 33),
    ("33+", 33, float("inf")),
)


@dataclass(frozen=True)
class SegmentError:
    """How the model performs on one slice of the data."""

    segment: str
    value: str
    n: int
    mae: float
    median_residual: float
    """Signed, in EUR. Away from zero means a systematic bias, not just noise."""

    mape: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "value": self.value,
            "n": self.n,
            "mae_eur": self.mae,
            "median_residual_eur": self.median_residual,
            "mape": self.mape,
        }


@dataclass(frozen=True)
class ErrorAnalysis:
    """A full error breakdown for one model on one set of rows."""

    overall: Metrics
    residuals: pd.DataFrame
    segments: list[SegmentError]
    worst_overpredictions: pd.DataFrame
    worst_underpredictions: pd.DataFrame

    def segments_for(self, segment: str) -> list[SegmentError]:
        return [s for s in self.segments if s.segment == segment]

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": {
                "mae_eur": self.overall.mae,
                "rmse_eur": self.overall.rmse,
                "r2": self.overall.r2,
                "mape": self.overall.mape,
                "n": self.overall.n,
            },
            "segments": [s.as_dict() for s in self.segments],
        }


def _band(values: pd.Series, bands: tuple[tuple[str, float, float], ...]) -> pd.Series:
    """Label each value with the band it falls in. Left-closed, right-open."""
    labels = pd.Series(pd.NA, index=values.index, dtype="object")
    for name, low, high in bands:
        labels[(values >= low) & (values < high)] = name
    return labels


def build_residuals(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_column: str,
) -> pd.DataFrame:
    """Attach predictions and residuals to the rows they belong to."""
    actual = frame[target_column].to_numpy(dtype=float)
    residual = predictions - actual

    # Age is present on every real training table; the fallback keeps this
    # usable on a frame that predates the feature without special-casing it
    # at every call site.
    age_band = (
        _band(frame["age"], AGE_BANDS)
        if "age" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="object")
    )

    # np.divide with `where`, not np.where: np.where evaluates both branches,
    # so the division still happens on the zero rows and only its result is
    # discarded — the guard silences nothing and numpy warns. This form skips
    # the division itself. The dataset has no zero label today; a division that
    # works only because of a property of today's data is tomorrow's bug.
    percentage_error = np.divide(
        np.abs(residual),
        actual,
        out=np.full_like(actual, np.nan, dtype=float),
        where=actual != 0,
    )

    return frame.assign(
        predicted=predictions,
        residual=residual,
        absolute_error=np.abs(residual),
        percentage_error=percentage_error,
        value_band=_band(frame[target_column], VALUE_BANDS),
        age_band=age_band,
    )


def segment_errors(
    residuals: pd.DataFrame,
    *,
    target_column: str,
    by: tuple[str, ...] = ("value_band", "age_band", "position", "season"),
    min_rows: int = 30,
) -> list[SegmentError]:
    """Error metrics per slice.

    Slices thinner than ``min_rows`` are dropped: an MAE over eleven players is
    a number people will quote and should not.
    """
    findings: list[SegmentError] = []

    for column in by:
        if column not in residuals.columns:
            continue
        for value, group in residuals.groupby(column, dropna=True, observed=True):
            if len(group) < min_rows:
                continue
            metrics = evaluate(group[target_column], group["predicted"])
            findings.append(
                SegmentError(
                    segment=column,
                    value=str(value),
                    n=len(group),
                    mae=metrics.mae,
                    median_residual=float(group["residual"].median()),
                    mape=metrics.mape,
                )
            )
    return findings


def analyse_errors(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_column: str,
    top_n: int = 15,
) -> ErrorAnalysis:
    """The whole breakdown: overall, per segment, and the individual misses."""
    residuals = build_residuals(frame, predictions, target_column=target_column)
    ordered = residuals.sort_values("residual")

    return ErrorAnalysis(
        overall=evaluate(frame[target_column], predictions),
        residuals=residuals,
        segments=segment_errors(residuals, target_column=target_column),
        # Signed, so these are genuinely the two opposite failure modes rather
        # than the same fifteen rows sorted by magnitude.
        worst_overpredictions=ordered.tail(top_n).iloc[::-1],
        worst_underpredictions=ordered.head(top_n),
    )
