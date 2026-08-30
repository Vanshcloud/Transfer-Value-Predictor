"""The model zoo: nine families behind one interface.

The brief asks for all nine. The risk with nine is that one of them is quietly
broken — a grid key that addresses nothing, an estimator that takes no seed, a
library that rejects the feature names the others accept — and nobody notices
because the other eight carry the report. These tests fit every family on a
small frame, so a broken one fails here rather than forty minutes into a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.build import TARGET_COLUMN
from src.models.registry import (
    MODEL_REGISTRY,
    PARAM_PREFIX,
    ModelSpec,
    build_pipeline,
)
from src.models.splits import RANDOM_SEED

NUMERIC = ["age", "goals", "minutes_played"]
CATEGORICAL = ["position", "country_of_citizenship"]

EXPECTED_FAMILIES = {
    "linear",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    # Phase 15: extremely randomised trees, and a ridge-blended stack of the
    # three boosters. Both named here rather than only in the registry, so
    # deleting a family is a two-file change someone has to mean.
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "stacked",
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    size = 300
    goals = rng.integers(0, 25, size)
    minutes = rng.integers(0, 3000, size)
    age = rng.uniform(18, 35, size)
    log_value = 13.0 + 0.09 * goals + 0.0004 * minutes - 0.03 * (age - 25) ** 2
    return pd.DataFrame(
        {
            "age": age,
            "goals": goals,
            "minutes_played": minutes,
            "position": rng.choice(["Attack", "Defender", "Midfield"], size),
            # A name carrying an apostrophe and an accent. LightGBM refuses to
            # train on feature names with special JSON characters, and one-hot
            # names inherit the category text, so this is the real shape of the
            # data rather than a contrived edge case.
            "country_of_citizenship": rng.choice(["Brazil", "Côte d'Ivoire", "Korea, South"], size),
            TARGET_COLUMN: np.expm1(log_value + rng.normal(0, 0.3, size)),
        }
    )


def test_the_registry_holds_every_family_the_brief_asks_for() -> None:
    assert set(MODEL_REGISTRY) == EXPECTED_FAMILIES


def test_registry_keys_match_their_spec_names() -> None:
    assert all(name == spec.name for name, spec in MODEL_REGISTRY.items())


@pytest.mark.parametrize("name", sorted(EXPECTED_FAMILIES))
class TestEveryFamily:
    def test_it_fits_and_predicts_in_euros(self, name: str, frame: pd.DataFrame) -> None:
        pipeline = build_pipeline(MODEL_REGISTRY[name], NUMERIC, CATEGORICAL)
        pipeline.fit(frame[NUMERIC + CATEGORICAL], frame[TARGET_COLUMN])
        predictions = pipeline.predict(frame[NUMERIC + CATEGORICAL])

        assert len(predictions) == len(frame)
        # Log-space predictions would sit around 13, not in the millions.
        assert np.median(predictions) > 100_000
        assert (predictions >= 0).all()

    def test_its_grid_addresses_real_parameters(self, name: str) -> None:
        """A grid key that addresses nothing fails silently in a manual search."""
        spec = MODEL_REGISTRY[name]
        pipeline = build_pipeline(spec, NUMERIC, CATEGORICAL)
        settable = pipeline.get_params(deep=True)

        for key in spec.prefixed_grid():
            assert key in settable, f"{name}: {key} addresses nothing"

    def test_setting_its_grid_values_actually_takes(self, name: str) -> None:
        spec = MODEL_REGISTRY[name]
        pipeline = build_pipeline(spec, NUMERIC, CATEGORICAL)
        for key, values in spec.prefixed_grid().items():
            pipeline.set_params(**{key: values[-1]})
            assert pipeline.get_params(deep=True)[key] == values[-1]

    def test_two_fits_agree(self, name: str, frame: pd.DataFrame) -> None:
        """Bit-identical, except where parallelism makes that impossible.

        random_forest averages over trees with ``n_jobs=-1`` and the reduction
        order is not fixed, so it agrees to about 1e-15 relative rather than
        exactly. The tolerance below is still nine orders of magnitude tighter
        than anything a missing seed could survive, which is what this test is
        actually guarding against.
        """
        features, target = frame[NUMERIC + CATEGORICAL], frame[TARGET_COLUMN]
        first = build_pipeline(MODEL_REGISTRY[name], NUMERIC, CATEGORICAL)
        second = build_pipeline(MODEL_REGISTRY[name], NUMERIC, CATEGORICAL)
        first.fit(features, target)
        second.fit(features, target)

        if name == "random_forest":
            np.testing.assert_allclose(first.predict(features), second.predict(features), rtol=1e-9)
        else:
            np.testing.assert_array_equal(first.predict(features), second.predict(features))


def test_grid_keys_use_the_pipeline_prefix() -> None:
    spec = ModelSpec("x", lambda: None, {"alpha": [1.0]})
    assert list(spec.prefixed_grid()) == [f"{PARAM_PREFIX}alpha"]


def test_a_family_without_a_grid_produces_an_empty_one() -> None:
    assert MODEL_REGISTRY["linear"].prefixed_grid() == {}


def test_feature_names_are_normalised_for_lightgbm(frame: pd.DataFrame) -> None:
    """The concrete failure: "Do not support special JSON characters"."""
    from src.models.baseline import build_preprocessor

    names = build_preprocessor(NUMERIC, CATEGORICAL).fit(frame).get_feature_names_out()
    assert all(str(name).replace("_", "").isalnum() for name in names)
