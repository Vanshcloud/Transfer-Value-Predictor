"""Which family ships, and the constraint that decides it.

The stacked ensemble scored best on both variants and does not ship. That is a
deliberate trade — under 2% of validation MAE against every prediction losing
its explanation — and a trade made in code is one that can be silently undone,
so it is asserted here.
"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import Metrics
from src.models.registry import (
    DEPLOYMENT_MEGABYTES,
    EXPLAINABLE_FAMILIES,
    MODEL_REGISTRY,
    UNEXPLAINABLE_FAMILIES,
)
from src.models.tuning import TuningResult
from src.pipelines.tune import FamilyResult, _select_winner


def result(name: str, mae: float, *, se: float = 0.0, size: int = 1000) -> FamilyResult:
    metrics = Metrics(mae=mae, rmse=mae * 2, r2=0.8, mape=0.5, n=100, mae_standard_error=se)
    return FamilyResult(
        tuning=TuningResult(model_name=name, best_params={}, cv_mae=mae, n_candidates=1),
        validation=metrics,
        fit_seconds=1.0,
        predict_seconds=0.1,
        size_bytes=size,
    )


class TestSelectWinner:
    def test_the_best_explainable_family_wins(self) -> None:
        winner = _select_winner(
            [result("lightgbm", 2_000_000), result("ridge", 3_000_000)], "performance_only"
        )
        assert winner.name == "lightgbm"

    def test_an_unexplainable_family_does_not_win_even_when_it_scores_best(self) -> None:
        """The measured case: stacked beat lightgbm by 1.92% and cannot say why
        it predicted anything. Every prediction response documents an
        explanation, so it is not eligible."""
        winner = _select_winner(
            [result("stacked", 1_900_000), result("lightgbm", 2_000_000)], "performance_only"
        )
        assert winner.name == "lightgbm"

    def test_the_margin_does_not_matter(self) -> None:
        """Explainability is a requirement, not a tiebreak. A model that cannot
        produce the documented field does not serve this API at any margin."""
        winner = _select_winner(
            [result("stacked", 1.0), result("lightgbm", 9_000_000)], "performance_only"
        )
        assert winner.name == "lightgbm"

    def test_it_falls_back_rather_than_shipping_nothing(self) -> None:
        """If somehow no explainable family ran, a model is better than none —
        and the caller finds out from the artifact, which records the family."""
        winner = _select_winner([result("stacked", 1_000_000)], "performance_only")
        assert winner.name == "stacked"

    def test_selection_is_still_by_validation_mae(self) -> None:
        winner = _select_winner(
            [result("lightgbm", 2_500_000), result("catboost", 2_400_000)], "performance_only"
        )
        assert winner.name == "catboost"


class TestOneStandardErrorRule:
    """Breiman et al. (1984): among models the data cannot tell apart, take the
    cheapest. The measured case that motivated it — XGBoost beating LightGBM by
    EUR 7,917 with a standard error of EUR 58,508, t = 0.37."""

    def test_a_gap_inside_one_standard_error_goes_to_the_cheaper_model(self) -> None:
        """The measured case: XGBoost nominally ahead by EUR 7,917, and 372 MB
        of image against LightGBM's 9."""
        winner = _select_winner(
            [
                result("xgboost", 2_055_713, se=58_508, size=1_290_000),
                result("lightgbm", 2_063_630, se=58_508, size=2_080_000),
            ],
            "performance_only",
        )
        assert winner.name == "lightgbm"

    def test_the_tiebreak_is_deployment_cost_not_artifact_size(self) -> None:
        """CatBoost serialises to a fifth of LightGBM's size and costs 328 MB of
        package. Artifact size would pick it; the image is what actually costs."""
        winner = _select_winner(
            [
                result("catboost", 2_067_628, se=58_508, size=370_000),
                result("lightgbm", 2_063_630, se=58_508, size=2_080_000),
            ],
            "performance_only",
        )
        assert winner.name == "lightgbm"

    def test_a_gap_outside_one_standard_error_is_a_real_win(self) -> None:
        """The rule must not become 'always take the smallest'."""
        winner = _select_winner(
            [
                result("xgboost", 2_000_000, se=10_000, size=9_000_000),
                result("lightgbm", 2_500_000, se=10_000, size=1_800_000),
            ],
            "performance_only",
        )
        assert winner.name == "xgboost"

    def test_the_boundary_is_inclusive(self) -> None:
        winner = _select_winner(
            [
                result("xgboost", 1_000_000, se=100_000, size=1_000_000),
                result("lightgbm", 1_100_000, se=100_000, size=1_800_000),
            ],
            "performance_only",
        )
        assert winner.name == "lightgbm"

    def test_a_much_worse_but_free_model_does_not_win(self) -> None:
        """Ridge costs 0 MB — it is pure scikit-learn — and is nowhere near."""
        winner = _select_winner(
            [
                result("lightgbm", 2_000_000, se=50_000, size=1_800_000),
                result("ridge", 4_000_000, se=50_000, size=30_000),
            ],
            "performance_only",
        )
        assert winner.name == "lightgbm"

    def test_with_no_standard_error_it_behaves_as_a_plain_minimum(self) -> None:
        """Artifacts written before Metrics carried an SE default it to 0."""
        winner = _select_winner(
            [
                result("xgboost", 2_055_713, size=1_290_000),
                result("lightgbm", 2_063_630, size=2_080_000),
            ],
            "performance_only",
        )
        assert winner.name == "xgboost"


class TestFamilyClassification:
    def test_every_registered_family_is_classified(self) -> None:
        assert set(MODEL_REGISTRY) == EXPLAINABLE_FAMILIES | UNEXPLAINABLE_FAMILIES

    def test_the_two_sets_do_not_overlap(self) -> None:
        assert not (EXPLAINABLE_FAMILIES & UNEXPLAINABLE_FAMILIES)

    @pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost", "ridge"])
    def test_the_families_that_expose_importances_are_explainable(self, family: str) -> None:
        assert family in EXPLAINABLE_FAMILIES

    def test_the_blend_is_not(self) -> None:
        """A StackingRegressor has neither feature_importances_ nor coef_. Its
        members do; the blend that combines them does not, and averaging theirs
        would describe a model that is not the one predicting."""
        assert "stacked" in UNEXPLAINABLE_FAMILIES


class TestDeploymentCost:
    def test_every_family_has_a_measured_cost(self) -> None:
        assert set(MODEL_REGISTRY) == set(DEPLOYMENT_MEGABYTES)

    def test_the_pure_sklearn_families_are_free(self) -> None:
        """scikit-learn is needed to unpickle any pipeline at all, so a family
        that adds nothing beyond it costs nothing to ship."""
        for family in ("ridge", "random_forest", "extra_trees", "gradient_boosting"):
            assert DEPLOYMENT_MEGABYTES[family] == 0

    def test_xgboost_carries_its_cuda_libraries(self) -> None:
        """81 MB of xgboost plus 291 MB of nvidia, measured in the built image."""
        assert DEPLOYMENT_MEGABYTES["xgboost"] > 300

    def test_lightgbm_is_the_cheap_booster(self) -> None:
        assert DEPLOYMENT_MEGABYTES["lightgbm"] < DEPLOYMENT_MEGABYTES["xgboost"]
        assert DEPLOYMENT_MEGABYTES["lightgbm"] < DEPLOYMENT_MEGABYTES["catboost"]
