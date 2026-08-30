"""Nine model families behind one interface.

The brief asks for all nine, so all nine are here — but they sit behind one
:class:`ModelSpec` and one pipeline builder, so adding a tenth is a dictionary
entry rather than a new code path. Each carries its own small grid; the grids
are deliberately tiny (see docs/EXPERIMENT_TRACKING.md section 6), because a
large search over nine families is hundreds of fits for a gain the temporal
split mostly washes out, and every extra configuration is another chance to
overfit the validation season.

Every estimator is wrapped so it fits on ``log1p(EUR)`` and predicts in EUR,
and every one is seeded from the single project constant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.models.baseline import build_preprocessor
from src.models.splits import RANDOM_SEED

# Grid keys address the estimator through the pipeline: the "model" step is a
# TransformedTargetRegressor, whose estimator lives at .regressor.
PARAM_PREFIX = "model__regressor__"


@dataclass(frozen=True)
class ModelSpec:
    """One model family, its constructor and the grid worth searching."""

    name: str
    factory: Callable[[], Any]
    grid: dict[str, list[Any]] = field(default_factory=dict)

    def prefixed_grid(self) -> dict[str, list[Any]]:
        """The grid, addressed through the pipeline."""
        return {f"{PARAM_PREFIX}{key}": values for key, values in self.grid.items()}


def build_pipeline(
    spec: ModelSpec,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    """Preprocessing and estimator as one object.

    One object matters beyond tidiness: a preprocessor fitted separately from
    the model is the classic way training and serving drift apart, because
    nothing then forces the two to travel together.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=spec.factory(), func=np.log1p, inverse_func=np.expm1
                ),
            ),
        ]
    )


def _linear() -> LinearRegression:
    return LinearRegression()


def _ridge() -> Ridge:
    return Ridge(random_state=RANDOM_SEED)


def _lasso() -> Lasso:
    # Higher max_iter than the default: on a one-hot matrix this wide the
    # coordinate descent does not converge in 1,000 and warns on every fit.
    return Lasso(random_state=RANDOM_SEED, max_iter=5000)


def _elastic_net() -> ElasticNet:
    return ElasticNet(random_state=RANDOM_SEED, max_iter=5000)


def _random_forest() -> RandomForestRegressor:
    """The one family whose predictions are not bit-reproducible.

    ``n_jobs=-1`` parallelises the average over trees, and the order that
    reduction happens in is not fixed, so two predictions from the *same*
    fitted forest differ by around 1e-15 relative. Measured on this data:
    23s per fit parallel against 101s single-threaded, which is 27 minutes for
    this family alone across the search. The trees themselves are seeded and
    identical either way.
    ponytail: parallel fit, last-bit prediction variance. Set n_jobs=1 if
    bit-exact predictions ever matter more than the search finishing.
    """
    return RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)


def _extra_trees() -> ExtraTreesRegressor:
    """Extremely randomised trees.

    Worth a slot beside RandomForest rather than instead of it: it splits on
    random thresholds rather than searching for the best one, which trades a
    little fit for a lot of variance reduction. On a panel where the target
    spans four orders of magnitude and a handful of rows carry enormous values,
    that trade sometimes wins. Same reproducibility caveat as RandomForest.
    """
    return ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1)


def _stacked() -> StackingRegressor:
    """The three boosters, blended by a ridge meta-learner.

    Stacking earns a place only if the families make *different* mistakes; if
    they agree, the blend is the same model at three times the cost. They do
    differ here — CatBoost's ordered boosting handles the high-cardinality
    citizenship column differently from LightGBM's leaf-wise growth — so it is
    worth measuring rather than assuming.

    ``cv=3`` and a ridge final estimator: the meta-learner sees out-of-fold
    predictions only, which is what stops it learning from a base model's own
    training fit. ``passthrough=False`` keeps it to blending three numbers
    rather than re-solving the problem.
    """
    return StackingRegressor(
        estimators=[
            ("lightgbm", _lightgbm()),
            ("xgboost", _xgboost()),
            ("catboost", _catboost()),
        ],
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_SEED),
        cv=3,
        n_jobs=1,
        passthrough=False,
    )


def _gradient_boosting() -> HistGradientBoostingRegressor:
    """scikit-learn's histogram gradient boosting.

    The histogram implementation rather than ``GradientBoostingRegressor``:
    the classic one is exact and single-threaded, and takes minutes per fit on
    37,000 rows where this takes about a second for indistinguishable accuracy.
    """
    return HistGradientBoostingRegressor(random_state=RANDOM_SEED)


def _xgboost() -> XGBRegressor:
    return XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)


def _lightgbm() -> LGBMRegressor:
    return LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)


def _catboost() -> CatBoostRegressor:
    """Tunable parameters are passed explicitly, at their defaults.

    CatBoost's scikit-learn wrapper reports only parameters that were passed to
    the constructor, and its ``set_params`` accepts an unknown key without
    complaint. Naming them here is what lets the registry test catch a typo in
    a grid key for this family the same way it does for the other eight.
    """
    return CatBoostRegressor(
        random_state=RANDOM_SEED,
        verbose=0,
        allow_writing_files=False,
        iterations=300,
        depth=6,
        learning_rate=0.05,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("linear", _linear),
        ModelSpec("ridge", _ridge, {"alpha": [0.1, 1.0, 10.0]}),
        ModelSpec("lasso", _lasso, {"alpha": [0.001, 0.01, 0.1]}),
        ModelSpec(
            "elastic_net",
            _elastic_net,
            {"alpha": [0.001, 0.01], "l1_ratio": [0.15, 0.5]},
        ),
        ModelSpec(
            "random_forest",
            _random_forest,
            {"n_estimators": [200], "min_samples_leaf": [1, 5]},
        ),
        ModelSpec(
            "extra_trees",
            _extra_trees,
            {"n_estimators": [200], "min_samples_leaf": [1, 5]},
        ),
        ModelSpec(
            "gradient_boosting",
            _gradient_boosting,
            {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]},
        ),
        ModelSpec(
            "xgboost",
            _xgboost,
            {"n_estimators": [300], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]},
        ),
        ModelSpec(
            "lightgbm",
            _lightgbm,
            {"n_estimators": [300], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
        ),
        ModelSpec(
            "catboost",
            _catboost,
            {"iterations": [300], "depth": [4, 6], "learning_rate": [0.05, 0.1]},
        ),
        # No grid. Each candidate refits three boosters over three folds, so a
        # two-point grid is nine extra booster fits per point for a blend whose
        # tuning lives in its members. It competes at its members' defaults.
        ModelSpec("stacked", _stacked),
    )
}
"""Every model the brief asks for, keyed by name. Insertion order is the order
they are trained and reported in."""

UNEXPLAINABLE_FAMILIES = frozenset({"stacked"})
"""Families that fit and predict but cannot say why.

``extract_feature_importance`` reads ``feature_importances_`` or ``coef_`` off
the fitted estimator. A ``StackingRegressor`` has neither: its members each
have one, but the blend that combines them does not, and averaging the members'
importances would describe a model that is not the one making the prediction.
SHAP has the same problem — TreeExplainer cannot walk a blend, and
KernelExplainer costs seconds per prediction against milliseconds.

Kept in the registry rather than removed. It is a legitimate model and the
leaderboard should show what it scores; it is simply not one this API can
ship, because every prediction response carries an explanation. The choice is
made in `src.pipelines.tune._select_winner`, where the cost is logged.
"""

EXPLAINABLE_FAMILIES = frozenset(MODEL_REGISTRY) - UNEXPLAINABLE_FAMILIES
"""Families whose fitted estimator exposes named importances, and which
therefore satisfy the API's documented `explanation` field."""

DEPLOYMENT_MEGABYTES: dict[str, int] = {
    # Measured inside the built API image, `du -sm` on site-packages:
    #   sklearn  56 MB — already required by every family, so it costs nothing
    #                    extra and is the baseline.
    #   lightgbm  9 MB
    #   xgboost  81 MB + 291 MB of nvidia CUDA libraries its manylinux wheel
    #            depends on, which a CPU inference path never opens
    #   catboost 269 MB + 59 MB of plotly, which it requires
    "linear": 0,
    "ridge": 0,
    "lasso": 0,
    "elastic_net": 0,
    "random_forest": 0,
    "extra_trees": 0,
    "gradient_boosting": 0,
    "lightgbm": 9,
    "xgboost": 372,
    "catboost": 328,
    "stacked": 372 + 328,
}
"""What shipping each family costs the serving image, in megabytes.

The tiebreak for the one-standard-error rule. The obvious proxy — the size of
the saved artifact — is the wrong one and points the wrong way: CatBoost
serialises to 0.37 MB against LightGBM's 2.08, which makes it look like the
cheap choice while dragging 328 MB of package behind it. A 1.7 MB difference in
a file that is mounted once is not a cost; 300 MB in every image pull is.

Zero means "already required": those families are pure scikit-learn, which the
service needs anyway to unpickle any pipeline at all.
"""
