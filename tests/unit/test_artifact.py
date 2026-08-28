"""The saved artifact, and the one guarantee that makes it worth saving.

A model file that cannot reproduce the numbers recorded beside it is worse than
no file: it looks authoritative and is not. The central test here saves, loads,
re-predicts and compares against the recorded metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate
from src.feature_engineering.build import TARGET_COLUMN
from src.models.artifact import (
    ARTIFACT_VERSION,
    ModelArtifact,
    extract_feature_importance,
    load,
    predict,
    save,
)
from src.models.registry import MODEL_REGISTRY, build_pipeline

NUMERIC = ["age", "goals"]
CATEGORICAL = ["position"]
FEATURES = ("age", "goals", "position")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    size = 200
    goals = rng.integers(0, 20, size)
    return pd.DataFrame(
        {
            "age": rng.uniform(18, 34, size),
            "goals": goals,
            "position": rng.choice(["Attack", "Defender"], size),
            TARGET_COLUMN: np.expm1(13 + 0.1 * goals + rng.normal(0, 0.2, size)),
        }
    )


def make_artifact(frame: pd.DataFrame, model_name: str = "ridge") -> ModelArtifact:
    pipeline = build_pipeline(MODEL_REGISTRY[model_name], NUMERIC, CATEGORICAL)
    pipeline.fit(frame[list(FEATURES)], frame[TARGET_COLUMN])
    metrics = evaluate(frame[TARGET_COLUMN], pipeline.predict(frame[list(FEATURES)]))

    return ModelArtifact(
        variant="performance_only",
        model_name=model_name,
        params={"model__regressor__alpha": 1.0},
        pipeline=pipeline,
        validation=metrics,
        test=metrics,
        feature_columns=FEATURES,
        target_column=TARGET_COLUMN,
        feature_importance=extract_feature_importance(pipeline),
        dataset={"table": "training_table", "rows": len(frame)},
        split={"strategy": "temporal"},
        seed=42,
        leaderboard=[{"model": model_name, "validation_mae_eur": metrics.mae}],
    )


class TestRoundTrip:
    def test_a_reloaded_artifact_reproduces_its_recorded_metrics_exactly(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        """The guarantee everything else rests on.

        Phase 7's verification, and the reason the fitted preprocessing lives
        inside the pipeline rather than beside it.
        """
        artifact = make_artifact(frame)
        reloaded = load(save(artifact, tmp_path))

        recomputed = evaluate(frame[TARGET_COLUMN], predict(reloaded, frame))
        assert recomputed == artifact.test

    def test_predictions_survive_the_round_trip_bit_for_bit(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        artifact = make_artifact(frame)
        before = predict(artifact, frame)
        after = predict(load(save(artifact, tmp_path)), frame)
        np.testing.assert_array_equal(before, after)

    def test_metadata_survives(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        reloaded = load(save(make_artifact(frame), tmp_path))
        assert reloaded.variant == "performance_only"
        assert reloaded.feature_columns == FEATURES
        assert reloaded.seed == 42
        assert reloaded.split["strategy"] == "temporal"

    def test_column_order_does_not_change_predictions(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        # predict() selects by the artifact's own column list, so a caller
        # handing over a differently ordered frame cannot silently shift inputs.
        artifact = load(save(make_artifact(frame), tmp_path))
        shuffled = frame[list(reversed(list(frame.columns)))]
        np.testing.assert_array_equal(predict(artifact, frame), predict(artifact, shuffled))


class TestSidecar:
    def test_a_readable_json_lands_beside_the_model(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        # "Which run was best?" should not require unpickling nine models.
        path = save(make_artifact(frame), tmp_path)
        sidecar = path.with_suffix(".json")
        assert sidecar.exists()

        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        assert metadata["model_name"] == "ridge"
        assert metadata["metrics"]["test"]["mae_eur"] > 0
        assert metadata["seed"] == 42

    def test_the_sidecar_records_every_tracked_field(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        """The contract in docs/EXPERIMENT_TRACKING.md, asserted."""
        metadata = json.loads(
            save(make_artifact(frame), tmp_path).with_suffix(".json").read_text(encoding="utf-8")
        )
        assert {
            "artifact_version",
            "created_at",
            "variant",
            "model_name",
            "params",
            "metrics",
            "feature_columns",
            "target_column",
            "feature_importance",
            "dataset",
            "split",
            "seed",
            "leaderboard",
        } <= set(metadata)

    def test_the_file_name_identifies_variant_and_model(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = save(make_artifact(frame), tmp_path)
        assert path.name == "performance_only__ridge.joblib"


class TestVersioning:
    def test_an_artifact_from_a_future_schema_is_refused(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        # An old artifact read by new code must fail loudly, not be misread.
        artifact = make_artifact(frame)
        artifact.artifact_version = ARTIFACT_VERSION + 1
        path = save(artifact, tmp_path)

        with pytest.raises(ValueError, match="artifact_version"):
            load(path)


class TestFeatureImportance:
    @pytest.mark.parametrize("model_name", ["ridge", "random_forest", "lightgbm"])
    def test_importances_are_named_after_preprocessing(
        self, model_name: str, frame: pd.DataFrame
    ) -> None:
        # Names come from the preprocessor, which is why it is configured with
        # set_output(transform="pandas"). Without it Phase 8's SHAP plots are a
        # wall of x0, x1, x2.
        artifact = make_artifact(frame, model_name)
        assert artifact.feature_importance
        assert any("goals" in name for name in artifact.feature_importance)

    def test_a_family_exposing_neither_coefficients_nor_importances_is_not_fatal(
        self, frame: pd.DataFrame
    ) -> None:
        pipeline = build_pipeline(MODEL_REGISTRY["ridge"], NUMERIC, CATEGORICAL)
        pipeline.fit(frame[list(FEATURES)], frame[TARGET_COLUMN])
        del pipeline.named_steps["model"].regressor_.coef_

        assert extract_feature_importance(pipeline) == {}
