"""Empirical prediction intervals, measured rather than assumed.

A gradient boosting regressor has no calibrated uncertainty. Printing
``"confidence": 0.87`` next to a prediction would be a fabricated number, and
this project does not ship those.

What can be measured is how wrong the model actually was on held-out seasons,
and that is what this module records: the quantiles of its own residuals, in
log space, split by the value band a prediction falls into. An interval built
this way says something true and checkable — "80% of held-out predictions in
this band landed between these bounds" — rather than something reassuring.

Residuals are taken in **log space** because that is where the model is
additive, and because a EUR 2M miss means something entirely different for a
EUR 500k player than for a EUR 90M one. Converted back, the interval is
multiplicative, which is the shape error in this data actually has.

Bands key on the **predicted** value, not the actual one. At serve time the
actual is exactly what nobody knows.

**Which rows the quantiles come from matters more than the method.** They are
measured on the *validation* season and then checked against the *test*
seasons, which the model and the interval have both never seen. Until the final
audit they were measured on the test seasons themselves, and an interval fitted
to the rows it is then scored on covers exactly its nominal level by
construction — 0.800 at a nominal 0.80, a number that is arithmetic rather than
evidence.

Measured properly, this method is still the best of the ones tried. Calibrating
on 2022 and evaluating on 2023 onward, at a nominal 80% (Winkler interval
score, lower better):

    method                                    coverage   log-width   Winkler
    band-wise quantiles (this module)            0.768       1.531   10.58M
    conformalised quantile regression (CQR)      0.776       1.710   10.98M
    split conformal, symmetric                   0.754       1.531   11.64M
    LightGBM quantile regression, unconformalised 0.626      1.294   12.53M

(performance-only; with-prior-value orders the same and reaches 0.798.)
Pooling several held-out seasons' residuals instead of one does not help —
0.761 to 0.768 across three variations — so the shortfall is not a sample-size
problem.

It is a temporal one. Split conformal guarantees coverage under
exchangeability, and consecutive football seasons are not exchangeable: the
market inflates, the panel's composition shifts. The shortfall from 0.80 to
0.77 is the size of that shift, and it is reported rather than closed, because
widening the interval until the number reads 0.80 would be fitting the test
set through the back door.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.error_analysis import VALUE_BANDS

DEFAULT_LEVEL = 0.8
"""Interval coverage. 80% rather than 95%: at 95% the bounds on this data are
so wide they stop informing anyone, and quoting a number nobody can act on is
its own kind of dishonesty."""

MIN_BAND_ROWS = 50
"""Below this, a band's own quantiles are noise and the overall ones are used
instead. The response says which was used."""

OVERALL_BAND = "overall"


def _band_for(value: float) -> str:
    for name, low, high in VALUE_BANDS:
        if low <= value < high:
            return name
    return OVERALL_BAND


def calibrate(
    actual: np.ndarray, predicted: np.ndarray, *, level: float = DEFAULT_LEVEL
) -> dict[str, Any]:
    """Measure residual quantiles per value band on held-out predictions.

    Args:
        actual: True values in EUR.
        predicted: Model predictions in EUR, from rows the model never trained on.
        level: Target coverage, e.g. 0.8 for an 80% interval.

    Returns:
        A JSON-able mapping the artifact carries and the API serves.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # log1p, matching the space the model fits in, so the interval scales with
    # the prediction instead of being a fixed number of euros everywhere.
    residual = np.log1p(actual) - np.log1p(predicted)
    tail = (1.0 - level) / 2.0

    bands: dict[str, Any] = {}
    labels = np.array([_band_for(value) for value in predicted])

    for name in [band for band, _, _ in VALUE_BANDS] + [OVERALL_BAND]:
        selected = residual if name == OVERALL_BAND else residual[labels == name]
        if len(selected) < (1 if name == OVERALL_BAND else MIN_BAND_ROWS):
            continue
        bands[name] = {
            "lower_log": float(np.quantile(selected, tail)),
            "upper_log": float(np.quantile(selected, 1.0 - tail)),
            "median_log": float(np.median(selected)),
            "n": int(len(selected)),
        }

    return {"level": level, "bands": bands}


def measured_coverage(
    calibration: dict[str, Any], actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    """Fraction of held-out rows the interval actually contained.

    The number that makes the calibration checkable. It is computed on rows the
    quantiles were *not* measured on — otherwise it returns the nominal level
    whatever the model does, which is what it used to do.

    Returns an empty mapping when there is nothing to measure, so a caller can
    store the result unconditionally without inventing a coverage of zero.
    """
    if not calibration.get("bands"):
        return {}

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not len(actual):
        return {}

    bounds = [interval_for(calibration, float(value)) for value in predicted]
    usable = np.array([bool(bound) for bound in bounds])
    if not usable.any():
        return {}

    lower = np.array([bound.get("lower_eur", np.nan) for bound in bounds])
    upper = np.array([bound.get("upper_eur", np.nan) for bound in bounds])
    inside = (actual >= lower) & (actual <= upper) & usable

    return {
        "level": float(calibration.get("level", DEFAULT_LEVEL)),
        "empirical": float(inside.sum() / usable.sum()),
        "n": int(usable.sum()),
        "median_width_eur": float(np.median((upper - lower)[usable])),
        "basis": (
            "measured on the test seasons, against quantiles taken from the "
            "validation season; neither the model nor the interval has seen these rows"
        ),
    }


def interval_for(calibration: dict[str, Any], prediction_eur: float) -> dict[str, Any]:
    """The interval around one prediction, and an honest note about its basis.

    Falls back to the overall band when the prediction's own band was measured
    over too few rows, and says so — an interval whose provenance is hidden
    invites more trust than it has earned.
    """
    bands = calibration.get("bands", {})
    if not bands:
        return {}

    band = _band_for(prediction_eur)
    used = band if band in bands else OVERALL_BAND
    if used not in bands:
        return {}

    entry = bands[used]
    logged = np.log1p(prediction_eur)
    basis = (
        f"empirical residual quantiles for the {used} band on held-out seasons"
        if used == band
        else (
            f"empirical residual quantiles across all held-out seasons; the "
            f"{band} band had too few rows to measure on its own"
        )
    )

    coverage = calibration.get("coverage") or {}
    return {
        "level": float(calibration.get("level", DEFAULT_LEVEL)),
        "lower_eur": float(np.expm1(logged + entry["lower_log"])),
        "upper_eur": float(np.expm1(logged + entry["upper_log"])),
        "basis": basis,
        "reference_rows": int(entry["n"]),
        # The level this interval actually achieved on seasons after the one it
        # was measured on, or None if it was never checked. Served next to the
        # nominal level rather than instead of it: a caller who sees only 0.8
        # will believe 0.8, and the honest figure is lower.
        "measured_coverage": (float(coverage["empirical"]) if "empirical" in coverage else None),
    }
