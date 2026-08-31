"""The honest number, established before the model zoo has anything to beat.

Two baselines: a regularised linear model and one gradient-boosting model. If
Phase 7's nine tuned models cannot beat these, the extra machinery is not
earning its place, and the only way to know that is to measure it now.

Every model is wrapped in :class:`~sklearn.compose.TransformedTargetRegressor`
so it fits on ``log1p(value)`` and predicts in EUR. The target's raw skew is
6.16 and falls to 0.37 under log1p; fitting on the raw value lets a handful of
200M outliers dominate the loss. Doing it through the wrapper rather than by
hand means no caller can forget the inverse transform, which is the usual way
this goes wrong.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.models.splits import RANDOM_SEED

# country_of_citizenship has ~180 levels with a long tail of single-player
# nations. Folding rare levels into one "infrequent" category keeps the matrix
# narrow and stops the model fitting a coefficient to a country it saw once.
MIN_CATEGORY_FREQUENCY = 20

# One-hot names inherit the category text, so country_of_citizenship produces
# columns like "Cote d'Ivoire". LightGBM refuses to train on a feature name
# containing special JSON characters ("Do not support special JSON characters
# in feature name"), so names are normalised once, here, rather than in each
# model that happens to care.
_UNSAFE_NAME = re.compile(r"[^0-9A-Za-z_]+")


def _safe_names(names: Sequence[str]) -> list[str]:
    return [_UNSAFE_NAME.sub("_", str(name)).strip("_") for name in names]


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = _safe_names(list(frame.columns))
    return renamed


def _renamed_feature_names(_transformer: object, input_features: Sequence[str]) -> np.ndarray:
    names = _safe_names(input_features)
    if len(set(names)) != len(names):
        # Two categories that differ only in punctuation would collide and make
        # feature importance ambiguous. Loud beats subtly wrong.
        raise ValueError("feature name collision after normalisation")
    return np.asarray(names, dtype=object)


def build_preprocessor(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> Pipeline:
    """Impute, scale and encode.

    ``sparse_output=False`` rather than ``sparse=``: the latter was removed in
    scikit-learn 1.9 and raises TypeError (plans/00-discovery.md section 3).
    ``set_output(transform="pandas")`` keeps feature names attached all the way
    through, which is what makes SHAP legible in Phase 8.
    """
    numeric = Pipeline(
        [
            # Height is 1.2% null and foot 1.7%; median imputation is the
            # boring choice and the one a baseline should make.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )

    columns = ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_features)),
            ("categorical", categorical, list(categorical_features)),
        ],
        remainder="drop",
    )
    preprocessor = Pipeline(
        [
            ("columns", columns),
            (
                "safe_names",
                FunctionTransformer(_rename_columns, feature_names_out=_renamed_feature_names),
            ),
        ]
    )
    return preprocessor.set_output(transform="pandas")


def _wrap(estimator: object) -> TransformedTargetRegressor:
    """Fit on log1p(EUR), predict in EUR."""
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)


def ridge_model(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> Pipeline:
    """A regularised linear baseline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", _wrap(Ridge(alpha=1.0, random_state=RANDOM_SEED))),
        ]
    )


def gradient_boosting_model(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> Pipeline:
    """A gradient-boosting baseline.

    scikit-learn's histogram implementation rather than XGBoost or LightGBM: it
    is already a dependency, needs no libomp, and this is a baseline whose job
    is to be beaten by Phase 7, not to win.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", _wrap(HistGradientBoostingRegressor(random_state=RANDOM_SEED))),
        ]
    )


BASELINE_MODELS: dict[str, Callable[[Sequence[str], Sequence[str]], Pipeline]] = {
    "ridge": ridge_model,
    "gradient_boosting": gradient_boosting_model,
}
