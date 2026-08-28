"""Baseline models and the training stage.

The trap this file guards is the target transform. The model fits on
log1p(EUR) and every reported number is in EUR, so an inverse transform that
goes missing produces predictions around 15 instead of around 5,000,000 — which
is obvious the moment anyone looks, and invisible in a metric that was also
computed in log space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.build import CATEGORICAL_FEATURES, TARGET_COLUMN
from src.models.baseline import (
    BASELINE_MODELS,
    build_preprocessor,
    gradient_boosting_model,
    ridge_model,
)
from src.models.splits import RANDOM_SEED

NUMERIC = ["age", "goals", "minutes_played"]
CATEGORICAL = ["position", "foot"]


@pytest.fixture
def training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    size = 400
    age = rng.uniform(18, 35, size)
    goals = rng.integers(0, 25, size)
    minutes = rng.integers(0, 3000, size)
    position = rng.choice(["Attack", "Defender", "Midfield"], size)

    # Built in log space, the way the real target behaves: a learnable signal
    # plus modest noise, which expm1 then turns into the four-orders-of-
    # magnitude skew the models actually meet.
    log_value = (
        13.0
        + 0.09 * goals
        + 0.0004 * minutes
        - 0.03 * (age - 25) ** 2
        + np.where(position == "Attack", 0.4, 0.0)
        + rng.normal(0, 0.3, size)
    )
    return pd.DataFrame(
        {
            "age": age,
            "goals": goals,
            "minutes_played": minutes,
            "position": position,
            "foot": rng.choice(["left", "right"], size),
            TARGET_COLUMN: np.expm1(log_value),
        }
    )


class TestPreprocessor:
    def test_it_keeps_feature_names_for_shap(self, training_frame: pd.DataFrame) -> None:
        # set_output(transform="pandas") is what makes Phase 8's SHAP plots
        # legible instead of a wall of x0, x1, x2.
        out = build_preprocessor(NUMERIC, CATEGORICAL).fit_transform(training_frame)
        assert isinstance(out, pd.DataFrame)
        assert any("position" in name for name in out.columns)

    def test_it_produces_a_dense_matrix(self, training_frame: pd.DataFrame) -> None:
        # sparse_output=False, because the old `sparse` argument to
        # OneHotEncoder was removed in scikit-learn 1.9 and raises TypeError.
        out = build_preprocessor(NUMERIC, CATEGORICAL).fit_transform(training_frame)
        assert out.to_numpy().dtype.kind == "f"

    def test_it_imputes_rather_than_dropping(self, training_frame: pd.DataFrame) -> None:
        frame = training_frame.copy()
        frame.loc[frame.index[:10], "age"] = np.nan
        frame.loc[frame.index[:10], "foot"] = None
        out = build_preprocessor(NUMERIC, CATEGORICAL).fit_transform(frame)
        assert len(out) == len(frame)
        assert not out.isna().to_numpy().any()

    def test_an_unseen_category_does_not_crash_prediction(
        self, training_frame: pd.DataFrame
    ) -> None:
        preprocessor = build_preprocessor(NUMERIC, CATEGORICAL).fit(training_frame)
        unseen = training_frame.head(1).copy()
        unseen["position"] = "Goalkeeper-Sweeper"
        assert len(preprocessor.transform(unseen)) == 1

    def test_columns_outside_the_two_lists_are_dropped(self, training_frame: pd.DataFrame) -> None:
        frame = training_frame.assign(secret=1.0)
        out = build_preprocessor(NUMERIC, CATEGORICAL).fit_transform(frame)
        assert not any("secret" in name for name in out.columns)


@pytest.mark.parametrize("factory", [ridge_model, gradient_boosting_model])
class TestBaselineModels:
    def test_predictions_come_back_in_euros_not_log_space(
        self, factory: object, training_frame: pd.DataFrame
    ) -> None:
        """The transform trap. Log-space predictions sit around 15, not millions."""
        model = factory(NUMERIC, CATEGORICAL)  # type: ignore[operator]
        model.fit(training_frame[NUMERIC + CATEGORICAL], training_frame[TARGET_COLUMN])
        predictions = model.predict(training_frame[NUMERIC + CATEGORICAL])

        assert predictions.min() > 1_000
        assert np.median(predictions) > 100_000

    def test_predictions_are_never_negative(
        self, factory: object, training_frame: pd.DataFrame
    ) -> None:
        # expm1 of anything real is > -1, and a market value cannot be negative.
        model = factory(NUMERIC, CATEGORICAL)  # type: ignore[operator]
        model.fit(training_frame[NUMERIC + CATEGORICAL], training_frame[TARGET_COLUMN])
        assert (model.predict(training_frame[NUMERIC + CATEGORICAL]) >= 0).all()

    def test_two_fits_agree_exactly(self, factory: object, training_frame: pd.DataFrame) -> None:
        features, target = training_frame[NUMERIC + CATEGORICAL], training_frame[TARGET_COLUMN]
        first = factory(NUMERIC, CATEGORICAL).fit(features, target)  # type: ignore[operator]
        second = factory(NUMERIC, CATEGORICAL).fit(features, target)  # type: ignore[operator]
        np.testing.assert_array_equal(first.predict(features), second.predict(features))

    def test_the_model_learns_something(
        self, factory: object, training_frame: pd.DataFrame
    ) -> None:
        from src.evaluation.metrics import evaluate

        features, target = training_frame[NUMERIC + CATEGORICAL], training_frame[TARGET_COLUMN]
        model = factory(NUMERIC, CATEGORICAL).fit(features, target)  # type: ignore[operator]
        # A fair bar rather than a flattering one: the synthetic signal is
        # quadratic in age and these baselines are handed only linear age, and
        # EUR-space R2 is dominated by the top of the market either way.
        assert evaluate(target, model.predict(features)).r2 > 0.3


def test_the_registry_exposes_both_baselines() -> None:
    assert set(BASELINE_MODELS) == {"ridge", "gradient_boosting"}


def test_the_prior_value_feature_is_supplied_in_log_space() -> None:
    """A raw-EUR lagged value against a log target is a misspecified linear model.

    Measured: raw, the column has skew 4.53 and its largest value sits 17
    standard deviations out; Ridge scored R^2 -1,773,318 because expm1 turned a
    large log-space prediction into billions. Tree models are invariant to
    monotone transforms and never showed the problem, which is exactly why it
    would have shipped.
    """
    from src.feature_engineering.build import PRIOR_VALUE_FEATURES

    assert "prev_log_market_value_in_eur" in PRIOR_VALUE_FEATURES
    assert "prev_market_value_in_eur" not in PRIOR_VALUE_FEATURES


def test_categorical_features_are_not_fed_to_the_numeric_branch() -> None:
    """The training stage derives its numeric list by subtraction; guard it."""
    from src.pipelines import train

    assert train.CATEGORICAL_FEATURES == CATEGORICAL_FEATURES
