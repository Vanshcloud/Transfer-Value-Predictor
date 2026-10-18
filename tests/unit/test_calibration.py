"""Prediction intervals: what they are fitted on, and what they may claim.

Audit F9. The quantile level is chosen from seasons before the test seasons,
and the coverage a response quotes is its value band's record, not the
overall one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration import (
    DEFAULT_LEVEL,
    QUANTILE_LEVELS,
    calibrate,
    interval_for,
    measured_coverage,
    select_quantile_level,
)

RNG = np.random.default_rng(3)


def season(spread: float, size: int = 4000, centre: float = 2e6) -> tuple[np.ndarray, np.ndarray]:
    """Actuals whose log residual against the prediction has this spread."""
    predicted = np.full(size, centre)
    actual = np.expm1(np.log1p(predicted) + RNG.normal(0.0, spread, size))
    return actual, predicted


class TestQuantileLevel:
    def test_stable_seasons_stay_at_or_just_above_the_nominal_level(self) -> None:
        """No drift, so nothing to correct beyond the finite-sample margin: an
        empirical 0.80 quantile covers a fraction under 80% of a fresh sample."""
        simulations = [(*season(0.5), [season(0.5), season(0.5)]) for _ in range(3)]
        chosen = select_quantile_level(simulations)
        assert DEFAULT_LEVEL <= chosen["level"] <= 0.815
        assert chosen["simulated_coverage_at_level"] >= DEFAULT_LEVEL

    def test_widening_seasons_widen_the_quantiles_until_the_target_is_met(self) -> None:
        """Later seasons spread more than the one calibrated on: the nominal
        quantiles undercover, and the rule picks the smallest level that
        reaches the target on those later seasons."""
        simulations = [(*season(0.5), [season(0.6), season(0.65)]) for _ in range(3)]
        chosen = select_quantile_level(simulations)
        assert chosen["simulated_coverage_at_nominal"] < DEFAULT_LEVEL
        assert chosen["simulated_coverage_at_level"] >= DEFAULT_LEVEL
        assert chosen["level"] > DEFAULT_LEVEL
        smaller = QUANTILE_LEVELS[QUANTILE_LEVELS.index(chosen["level"]) - 1]
        assert smaller >= DEFAULT_LEVEL  # the chosen level is the smallest that works
        assert chosen["measurements"] == 6

    def test_the_nominal_level_is_what_a_response_reports(self) -> None:
        actual, predicted = season(0.5)
        calibration = calibrate(actual, predicted, quantile_level=0.85)
        assert calibration["level"] == DEFAULT_LEVEL
        assert calibration["quantile_level"] == 0.85
        assert interval_for(calibration, 2e6)["level"] == DEFAULT_LEVEL

    def test_wider_quantiles_give_a_wider_interval(self) -> None:
        actual, predicted = season(0.5)
        narrow = interval_for(calibrate(actual, predicted), 2e6)
        wide = interval_for(calibrate(actual, predicted, quantile_level=0.9), 2e6)
        assert wide["lower_eur"] < narrow["lower_eur"] < narrow["upper_eur"] < wide["upper_eur"]


class TestCoverageByBand:
    def test_each_band_reports_its_own_coverage_and_rows(self) -> None:
        low_actual, low_predicted = season(0.5, size=3000, centre=5e5)
        high_actual, high_predicted = season(0.5, size=1000, centre=3e7)
        actual = np.concatenate([low_actual, high_actual])
        predicted = np.concatenate([low_predicted, high_predicted])
        calibration = calibrate(actual, predicted)

        # Later: the low band holds, the high band spreads out.
        later_low = season(0.5, size=3000, centre=5e5)
        later_high = season(1.5, size=1000, centre=3e7)
        coverage = measured_coverage(
            calibration,
            np.concatenate([later_low[0], later_high[0]]),
            np.concatenate([later_low[1], later_high[1]]),
        )
        assert coverage["by_band"]["<1M"]["n"] == 3000
        assert coverage["by_band"]["20M-50M"]["n"] == 1000
        assert coverage["by_band"]["20M-50M"]["empirical"] < coverage["by_band"]["<1M"]["empirical"]

        calibration["coverage"] = coverage
        served = interval_for(calibration, 3e7)
        assert served["band"] == "20M-50M"
        assert served["band_measured_coverage"] == coverage["by_band"]["20M-50M"]["empirical"]
        assert served["band_coverage_rows"] == 1000
        # The overall figure is still served, and is not the band's.
        assert served["measured_coverage"] == coverage["empirical"]
        assert served["measured_coverage"] != served["band_measured_coverage"]

    def test_an_unmeasured_band_says_so_rather_than_borrowing_another(self) -> None:
        actual, predicted = season(0.5, centre=5e5)
        calibration = calibrate(actual, predicted)
        calibration["coverage"] = measured_coverage(calibration, actual, predicted)
        served = interval_for(calibration, 8e7)  # no test predictions above 50M
        assert served["band"] == ">50M"
        assert served["band_measured_coverage"] is None
        assert served["band_coverage_rows"] is None
