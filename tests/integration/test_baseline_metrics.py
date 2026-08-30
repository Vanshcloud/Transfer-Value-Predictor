"""The baseline table, measured on the full dataset.

Marked ``integration``: needs ``scripts/fetch_data.py`` and
``scripts/build_features.py`` to have run. Skips rather than fails otherwise.

These bounds are wide. They are a regression fence, not a specification — their
job is to catch a split that silently stopped splitting or a target transform
that silently stopped inverting, both of which move a metric by an order of
magnitude rather than a few points.
"""

from __future__ import annotations

import pytest

from src.pipelines.features import TRAINING_TABLE
from src.pipelines.train import BaselineResult, render_comparison, run_baselines
from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.config import load_settings
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def results() -> list[BaselineResult]:
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    if not store.has_table(TRAINING_TABLE):
        pytest.skip("run scripts/build_features.py first")
    return run_baselines(store, load_settings().split)


def find(results: list[BaselineResult], variant: str, split: str, model: str) -> BaselineResult:
    return next(r for r in results if (r.variant, r.split, r.model) == (variant, split, model))


def test_every_variant_split_and_model_combination_ran(
    results: list[BaselineResult],
) -> None:
    assert len(results) == 2 * 3 * 2


def test_the_temporal_baseline_is_where_phase_15_left_it(
    results: list[BaselineResult],
) -> None:
    """R2 0.41 -> 0.76 and MAE EUR 4.5M -> 2.4M.

    The spike's 0.412 was measured on 19 features and a 120-day label window.
    Widening the window to a year and adding the context and role features from
    the six Kaggle files the project was not downloading moved both. Wide
    tolerances because this is an untuned baseline on a weekly-refreshed
    dataset, not a headline metric — the point is to catch a collapse, not to
    pin a decimal.
    """
    result = find(results, "performance_only", "temporal", "gradient_boosting")
    assert result.test.r2 == pytest.approx(0.76, abs=0.12)
    assert result.test.mae == pytest.approx(2_400_000, rel=0.30)


def test_prior_value_lifts_r2_the_way_the_spike_measured(
    results: list[BaselineResult],
) -> None:
    """R2 0.45 -> 0.83 was the finding that forced shipping two variants.

    The gap has narrowed — 0.76 -> 0.90 rather than 0.45 -> 0.83 — because the
    performance-only model got most of what the prior value used to supply on
    its own: league strength, club results and squad role. That the prior value
    still helps is the point; that it helps less is the improvement.
    """
    without = find(results, "performance_only", "temporal", "gradient_boosting")
    with_prior = find(results, "with_prior_value", "temporal", "gradient_boosting")
    assert with_prior.test.r2 > without.test.r2 + 0.05


def test_temporal_is_harder_than_random(results: list[BaselineResult]) -> None:
    """The spike's central finding: temporal EUR MAE is far worse.

    If this ever inverts, the temporal split has stopped being temporal.
    """
    temporal = find(results, "performance_only", "temporal", "gradient_boosting")
    randomised = find(results, "performance_only", "random", "gradient_boosting")
    assert temporal.test.mae > randomised.test.mae


def test_no_model_collapses(results: list[BaselineResult]) -> None:
    """A negative R2 means the model is worse than predicting the mean.

    This is the fence that would have caught the raw-EUR prior value: Ridge
    scored R2 -1,773,318 on it before the feature was moved to log space.
    """
    for result in results:
        assert result.test.r2 > 0.0, f"{result.variant}/{result.split}/{result.model}"
        assert result.test.mae < 50_000_000


def test_metrics_are_in_euros_not_log_space(results: list[BaselineResult]) -> None:
    # A missing inverse transform puts MAE around 0.5 rather than millions.
    for result in results:
        assert result.test.mae > 100_000


def test_the_run_is_reproducible(results: list[BaselineResult]) -> None:
    """Two runs must agree exactly, not approximately."""
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    assert run_baselines(store, load_settings().split) == results


def test_the_comparison_marks_the_headline(results: list[BaselineResult]) -> None:
    rendered = render_comparison(results)
    assert "temporal" in rendered
    assert "headline" in rendered
    assert sum(r.is_headline for r in results) == 4
