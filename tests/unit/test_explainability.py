"""Explanations as data.

The contract these tests pin down is that an explanation is a structure, not a
picture: Phase 9 returns it as JSON and Phase 10 draws it in a browser, so
anything that only exists inside a matplotlib figure is unusable to both.

The other thing under test is the log-space unit. The model fits log1p(EUR), so
SHAP values are additive in log space and not in euros. Getting that wrong
produces an explanation that looks plausible and is arithmetically false.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate
from src.explainability.shap_explainer import (
    Contribution,
    explain_global,
    explain_prediction,
    supports_shap,
)
from src.feature_engineering.build import TARGET_COLUMN
from src.models.artifact import ModelArtifact, extract_feature_importance
from src.models.registry import MODEL_REGISTRY, build_pipeline

NUMERIC = ["age", "goals", "minutes_played"]
CATEGORICAL = ["position"]
FEATURES = ("age", "goals", "minutes_played", "position")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    size = 300
    goals = rng.integers(0, 25, size)
    minutes = rng.integers(0, 3000, size)
    age = rng.uniform(18, 35, size)
    log_value = 13.0 + 0.11 * goals + 0.0004 * minutes - 0.03 * (age - 25) ** 2
    return pd.DataFrame(
        {
            "age": age,
            "goals": goals,
            "minutes_played": minutes,
            "position": rng.choice(["Attack", "Defender", "Midfield"], size),
            TARGET_COLUMN: np.expm1(log_value + rng.normal(0, 0.25, size)),
        }
    )


def build(frame: pd.DataFrame, model_name: str = "lightgbm") -> ModelArtifact:
    pipeline = build_pipeline(MODEL_REGISTRY[model_name], NUMERIC, CATEGORICAL)
    pipeline.fit(frame[list(FEATURES)], frame[TARGET_COLUMN])
    metrics = evaluate(frame[TARGET_COLUMN], pipeline.predict(frame[list(FEATURES)]))
    return ModelArtifact(
        variant="performance_only",
        model_name=model_name,
        params={},
        pipeline=pipeline,
        validation=metrics,
        test=metrics,
        feature_columns=FEATURES,
        target_column=TARGET_COLUMN,
        feature_importance=extract_feature_importance(pipeline),
        dataset={"rows": len(frame)},
        split={"strategy": "temporal"},
        seed=42,
    )


@pytest.fixture(scope="module")
def artifact(frame: pd.DataFrame) -> ModelArtifact:
    return build(frame)


class TestSupportsShap:
    def test_a_tree_model_is_supported(self, artifact: ModelArtifact) -> None:
        assert supports_shap(artifact)

    def test_a_linear_model_is_reported_rather_than_crashing(self, frame: pd.DataFrame) -> None:
        # A KernelExplainer fallback nobody would wait for is worse than
        # saying so and showing coefficients.
        assert not supports_shap(build(frame, "ridge"))


class TestGlobalExplanation:
    def test_it_ranks_the_feature_the_data_was_built_around(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        # The synthetic target leans hardest on goals, so a working explainer
        # must put it near the top. This is what catches an explainer wired to
        # the wrong estimator.
        top = [name for name, _ in explain_global(artifact, frame).ranked(3)]
        assert any("goals" in name for name in top)

    def test_it_covers_every_preprocessed_feature(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        explanation = explain_global(artifact, frame)
        assert set(explanation.mean_abs_shap) == set(explanation.features)
        assert len(explanation.features) > len(NUMERIC)  # one-hot expansion

    def test_magnitudes_are_never_negative(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        assert all(v >= 0 for v in explain_global(artifact, frame).mean_abs_shap.values())

    def test_it_samples_large_frames(self, artifact: ModelArtifact, frame: pd.DataFrame) -> None:
        assert explain_global(artifact, frame, sample_size=50).sample_size == 50

    def test_sampling_is_seeded(self, artifact: ModelArtifact, frame: pd.DataFrame) -> None:
        first = explain_global(artifact, frame, sample_size=50).mean_abs_shap
        second = explain_global(artifact, frame, sample_size=50).mean_abs_shap
        assert first == second

    def test_it_serialises_to_json(self, artifact: ModelArtifact, frame: pd.DataFrame) -> None:
        """np.float64 is not JSON-serialisable; the API would 500 on it."""
        payload = explain_global(artifact, frame, sample_size=50).as_dict()
        assert json.loads(json.dumps(payload))["features"]


class TestPredictionExplanation:
    def test_contributions_sum_to_the_prediction_in_log_space(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        """SHAP's additivity property, which is the whole guarantee.

        It holds in log space because that is where the model predicts. It does
        NOT hold in euros, and this test is what stops someone "fixing" the
        module to report euro contributions.
        """
        row = frame.head(1)
        explanation = explain_prediction(artifact, row)

        total = np.log1p(explanation.base_value_eur) + sum(
            c.shap_value for c in explanation.contributions
        )
        assert total == pytest.approx(np.log1p(explanation.prediction_eur), rel=1e-6)

    def test_the_multiplier_is_the_exponent_of_the_contribution(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        for contribution in explain_prediction(artifact, frame.head(1)).contributions:
            assert contribution.effect_multiplier == pytest.approx(np.exp(contribution.shap_value))

    def test_the_prediction_matches_the_pipeline(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        row = frame.head(1)
        expected = artifact.pipeline.predict(row[list(FEATURES)])[0]
        assert explain_prediction(artifact, row).prediction_eur == pytest.approx(expected)

    def test_positive_and_negative_contributions_are_separated(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        # Phase 9 returns these as "what raised the value" and "what lowered it".
        explanation = explain_prediction(artifact, frame.head(1))
        assert all(c.shap_value > 0 for c in explanation.positive())
        assert all(c.shap_value < 0 for c in explanation.negative())
        assert len(explanation.positive()) + len(explanation.negative()) == len(
            explanation.contributions
        )

    def test_top_is_ordered_by_magnitude_regardless_of_sign(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        top = explain_prediction(artifact, frame.head(1)).top(5)
        magnitudes = [abs(c.shap_value) for c in top]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_it_refuses_anything_but_one_row(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        # Silently explaining the first of five rows is a confident wrong answer.
        with pytest.raises(ValueError, match="one row"):
            explain_prediction(artifact, frame.head(5))

    def test_column_order_does_not_change_the_explanation(
        self, artifact: ModelArtifact, frame: pd.DataFrame
    ) -> None:
        row = frame.head(1)
        shuffled = row[list(reversed(list(row.columns)))]
        assert (
            explain_prediction(artifact, row).as_dict()
            == explain_prediction(artifact, shuffled).as_dict()
        )

    def test_it_serialises_to_json(self, artifact: ModelArtifact, frame: pd.DataFrame) -> None:
        payload = explain_prediction(artifact, frame.head(1)).as_dict()
        restored = json.loads(json.dumps(payload))
        assert restored["contributions"]
        assert {"feature", "value", "shap_value", "effect_multiplier", "direction"} == set(
            restored["contributions"][0]
        )


def test_direction_reads_in_plain_english() -> None:
    assert Contribution("age", 30, 0.4, 1.49).direction == "increases"
    assert Contribution("age", 30, -0.4, 0.67).direction == "decreases"
