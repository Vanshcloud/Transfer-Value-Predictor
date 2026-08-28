"""Metrics, in EUR.

Reporting in log space is the quiet way to flatter a model: log compresses
exactly the expensive mistakes at the top of the market, where being wrong by
EUR 40M looks the same as being wrong by EUR 400k lower down.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import evaluate


def test_a_perfect_prediction_scores_perfectly() -> None:
    truth = np.array([1e6, 5e6, 2e7])
    metrics = evaluate(truth, truth)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.r2 == 1.0
    assert metrics.mape == 0.0


def test_mae_is_the_mean_absolute_error_in_eur() -> None:
    metrics = evaluate([1_000_000, 2_000_000], [1_500_000, 2_500_000])
    assert metrics.mae == 500_000


def test_rmse_punishes_the_large_miss_harder_than_mae() -> None:
    metrics = evaluate([1e6, 1e6], [1e6, 5e6])
    assert metrics.rmse > metrics.mae


def test_r2_is_zero_for_a_model_that_only_predicts_the_mean() -> None:
    truth = np.array([1e6, 3e6, 5e6])
    assert evaluate(truth, np.full(3, truth.mean())).r2 == pytest.approx(0.0)


def test_n_records_how_many_rows_were_scored() -> None:
    assert evaluate([1, 2, 3], [1, 2, 3]).n == 3


def test_metrics_render_in_readable_euros() -> None:
    rendered = evaluate([1_000_000, 2_000_000], [1_500_000, 2_500_000]).render()
    assert "500,000" in rendered
    assert "MAE EUR" in rendered


def test_evaluate_accepts_lists_and_series_alike() -> None:
    import pandas as pd

    truth = [1e6, 2e6]
    assert evaluate(truth, truth) == evaluate(pd.Series(truth), np.array(truth))
