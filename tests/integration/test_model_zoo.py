"""The zoo on real data, end to end.

Marked ``integration``: needs the training table. Runs a **subset** of families
by default — the point is that the machinery works on real data, and paying
twenty minutes of CatBoost and RandomForest search to learn that would mean
nobody runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.metrics import evaluate
from src.feature_engineering.build import TARGET_COLUMN, select_variant
from src.models.artifact import load, predict, save
from src.models.splits import RANDOM_SEED, temporal_split
from src.pipelines.features import TRAINING_TABLE
from src.pipelines.tune import render_leaderboard, train_variant
from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.config import load_settings
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration

FAMILIES = ("linear", "ridge", "lightgbm")


@pytest.fixture(scope="module")
def artifact() -> object:
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    if not store.has_table(TRAINING_TABLE):
        pytest.skip("run scripts/build_features.py first")

    settings = load_settings()
    return train_variant(
        store.read_table(TRAINING_TABLE),
        "performance_only",
        include_prior_value=False,
        config=settings.split,
        model_names=FAMILIES,
    )


def test_every_requested_family_reaches_the_leaderboard(artifact: object) -> None:
    assert {row["model"] for row in artifact.leaderboard} == set(FAMILIES)  # type: ignore[attr-defined]


def test_the_leaderboard_is_ordered_by_the_selection_metric(artifact: object) -> None:
    scores = [row["validation_mae_eur"] for row in artifact.leaderboard]  # type: ignore[attr-defined]
    assert scores == sorted(scores)


def test_the_winner_is_the_best_validation_mae(artifact: object) -> None:
    """Selection is by validation MAE in EUR, never by test and never by R2."""
    best = artifact.leaderboard[0]  # type: ignore[attr-defined]
    assert artifact.model_name == best["model"]  # type: ignore[attr-defined]
    assert artifact.validation.mae == pytest.approx(best["validation_mae_eur"])  # type: ignore[attr-defined]


def test_the_artifact_records_the_full_tracking_contract(artifact: object) -> None:
    metadata = artifact.metadata()  # type: ignore[attr-defined]
    assert metadata["seed"] == RANDOM_SEED
    assert metadata["split"]["strategy"] == "temporal"
    assert metadata["dataset"]["rows"] > 30_000
    assert metadata["feature_columns"]
    assert metadata["feature_importance"]


def test_metrics_are_in_euros_and_the_model_is_not_broken(artifact: object) -> None:
    assert artifact.test.mae > 100_000  # type: ignore[attr-defined]
    assert artifact.test.r2 > 0.0  # type: ignore[attr-defined]


def test_the_saved_artifact_reproduces_its_test_metrics_exactly(
    artifact: object, tmp_path: Path
) -> None:
    """Phase 7's headline verification, on the real table.

    Anything less means the recorded numbers describe a model that no longer
    exists, which is worse than recording nothing.
    """
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    settings = load_settings()
    frame, _ = select_variant(store.read_table(TRAINING_TABLE), include_prior_value=False)
    split = temporal_split(
        frame,
        train_end_season=settings.split.train_end_season,
        validation_season=settings.split.validation_season,
        test_start_season=settings.split.test_start_season,
    )

    reloaded = load(save(artifact, tmp_path))  # type: ignore[arg-type]
    test_rows = frame.loc[split.test]
    recomputed = evaluate(test_rows[TARGET_COLUMN], predict(reloaded, test_rows))

    assert recomputed == artifact.test  # type: ignore[attr-defined]


def test_the_selection_is_reproducible(artifact: object) -> None:
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    settings = load_settings()
    again = train_variant(
        store.read_table(TRAINING_TABLE),
        "performance_only",
        include_prior_value=False,
        config=settings.split,
        model_names=FAMILIES,
    )
    assert again.model_name == artifact.model_name  # type: ignore[attr-defined]
    assert again.params == artifact.params  # type: ignore[attr-defined]
    assert again.test == artifact.test  # type: ignore[attr-defined]


def test_the_leaderboard_renders_and_marks_the_winner(artifact: object) -> None:
    rendered = render_leaderboard(artifact)  # type: ignore[arg-type]
    assert "selected" in rendered
    assert artifact.model_name in rendered  # type: ignore[attr-defined]
