"""Regression metrics, reported in EUR.

The model trains on ``log1p`` of the target because the raw skew is 8.70. Every
number here is computed *after* the inverse transform, in euros, because "0.31
RMSE in log space" is not a quantity anyone can act on and it silently flatters
the model: log space compresses exactly the expensive mistakes at the top of
the market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


@dataclass(frozen=True)
class Metrics:
    """One model's performance on one set of rows, in EUR."""

    mae: float
    rmse: float
    r2: float
    mape: float
    n: int

    def render(self) -> str:
        return (
            f"MAE EUR {self.mae:>12,.0f}  RMSE EUR {self.rmse:>12,.0f}  "
            f"R2 {self.r2:>6.3f}  MAPE {self.mape:>6.1%}  n {self.n:>6,}"
        )


def evaluate(y_true: np.ndarray | object, y_pred: np.ndarray | object) -> Metrics:
    """Score predictions against truth, both in EUR.

    ``root_mean_squared_error`` rather than the old ``squared`` argument to
    ``mean_squared_error``: that argument was removed in scikit-learn 1.9 and
    now raises TypeError rather than warning (plans/00-discovery.md section 3).
    """
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    return Metrics(
        mae=float(mean_absolute_error(truth, predicted)),
        rmse=float(root_mean_squared_error(truth, predicted)),
        r2=float(r2_score(truth, predicted)),
        mape=float(mean_absolute_percentage_error(truth, predicted)),
        n=len(truth),
    )
