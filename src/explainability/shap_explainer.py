"""Explanations as data, not as pictures.

Every function here returns a plain structure — dataclasses and floats — and
nothing here imports matplotlib. That is deliberate: Phase 9's ``POST /predict``
has to return per-prediction contributions as JSON, and Phase 10 has to draw
them in a browser. If the only way to explain a prediction were a matplotlib
figure, both would have to reimplement the explanation and the three copies
would drift. Rendering lives in :mod:`src.evaluation.reports`, on top of this.

**Contributions are in log space, and that is not a detail to gloss over.**
The models fit ``log1p(EUR)``, so SHAP values are additive in log space and
emphatically *not* additive in euros: you cannot say "age contributed
-EUR 2M" because the same log contribution is worth a different number of
euros for a EUR 500k player and a EUR 90M one. What *is* exact is the
multiplicative reading — a log contribution of 0.34 multiplies the prediction
by ``exp(0.34)`` = 1.41, whatever the player is worth — so each contribution
carries an ``effect_multiplier`` alongside its raw value. That is the honest
unit for a log-target model, and it is the one the API should surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.models.artifact import ModelArtifact
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_SIZE = 2_000
"""Rows sampled for a global explanation.

SHAP over the whole test set is exact and slow; over a seeded sample it is
stable to well within the precision anyone reads off a bar chart. Raise it if
a ranking ever looks unstable between runs.
"""


@dataclass(frozen=True)
class Contribution:
    """One feature's push on one prediction."""

    feature: str
    value: Any
    """The feature's value for this row, so the explanation reads on its own."""

    shap_value: float
    """Additive contribution in log space."""

    effect_multiplier: float
    """``exp(shap_value)``: what this feature multiplied the prediction by.

    Above 1.0 pushed the value up, below 1.0 pushed it down. Exact, unlike any
    attempt to express the same thing in euros.
    """

    @property
    def direction(self) -> str:
        return "increases" if self.shap_value > 0 else "decreases"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": _plain(self.value),
            "shap_value": self.shap_value,
            "effect_multiplier": self.effect_multiplier,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class PredictionExplanation:
    """Why one player got the prediction they got."""

    prediction_eur: float
    base_value_eur: float
    """What the model predicts before any feature is considered: the training
    mean in log space, mapped back to EUR."""

    contributions: list[Contribution]

    def top(self, n: int = 10) -> list[Contribution]:
        """The n features that moved this prediction most, either direction."""
        return sorted(self.contributions, key=lambda c: -abs(c.shap_value))[:n]

    def positive(self) -> list[Contribution]:
        return [c for c in self.top(len(self.contributions)) if c.shap_value > 0]

    def negative(self) -> list[Contribution]:
        return [c for c in self.top(len(self.contributions)) if c.shap_value < 0]

    def as_dict(self, top_n: int = 10) -> dict[str, Any]:
        """JSON-ready. This is the shape Phase 9's /predict response returns."""
        return {
            "prediction_eur": self.prediction_eur,
            "base_value_eur": self.base_value_eur,
            "contributions": [c.as_dict() for c in self.top(top_n)],
        }


@dataclass(frozen=True)
class GlobalExplanation:
    """What the model relies on across many predictions."""

    features: list[str]
    mean_abs_shap: dict[str, float]
    """Average absolute log-space contribution: how much a feature moves
    predictions, ignoring which way."""

    mean_shap: dict[str, float]
    """Average signed contribution: which way it usually pushes."""

    sample_size: int
    shap_values: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    sample: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    """Retained so a beeswarm can be drawn without recomputing SHAP."""

    def ranked(self, n: int | None = None) -> list[tuple[str, float]]:
        ordered = sorted(self.mean_abs_shap.items(), key=lambda kv: -kv[1])
        return ordered[:n] if n else ordered

    def as_dict(self, top_n: int = 20) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "features": [
                {
                    "feature": name,
                    "mean_abs_shap": value,
                    "mean_shap": self.mean_shap[name],
                }
                for name, value in self.ranked(top_n)
            ],
        }


def _plain(value: object) -> object:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _preprocess(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Run the fitted preprocessing, keeping feature names.

    Names survive because the preprocessor is built with
    ``set_output(transform="pandas")``. Without that, every explanation in this
    module would be a ranking of x0, x1, x2.
    """
    transformed: pd.DataFrame = pipeline.named_steps["preprocess"].transform(frame)
    return transformed


def _tree_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """A TreeExplainer over the estimator *inside* the target transform.

    The pipeline's model step is a ``TransformedTargetRegressor``; SHAP must see
    the regressor it wraps, which is the thing that actually predicts in log
    space. Explaining the wrapper would explain the inverse transform too and
    the values would no longer be additive.
    """
    return shap.TreeExplainer(pipeline.named_steps["model"].regressor_)


def supports_shap(artifact: ModelArtifact) -> bool:
    """Whether this artifact's model is one TreeExplainer can handle.

    Linear families are not tree models. Rather than fall back to a slow
    KernelExplainer nobody would wait for, the report says so and shows
    coefficients instead.
    """
    try:
        _tree_explainer(artifact.pipeline)
    except Exception:  # noqa: BLE001 - shap raises a variety of types here
        return False
    return True


def explain_global(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
) -> GlobalExplanation:
    """Rank features by how much they move predictions across a sample."""
    rows = frame
    if len(frame) > sample_size:
        rows = frame.sample(sample_size, random_state=seed)

    features = _preprocess(artifact.pipeline, rows[list(artifact.feature_columns)])
    explainer = _tree_explainer(artifact.pipeline)

    # explainer(X) returns an Explanation; .shap_values() returns a bare
    # ndarray and is the legacy path (plans/00-discovery.md section 3).
    explanation = explainer(features)
    values = np.asarray(explanation.values)

    names = [str(name) for name in features.columns]
    logger.info("computed SHAP over %d rows, %d features", len(features), len(names))

    return GlobalExplanation(
        features=names,
        # float(), not the numpy scalar: these go straight into a JSON
        # response in Phase 9, and json.dump cannot serialise np.float64.
        mean_abs_shap={
            name: float(value)
            for name, value in zip(names, np.abs(values).mean(axis=0), strict=True)
        },
        mean_shap={
            name: float(value) for name, value in zip(names, values.mean(axis=0), strict=True)
        },
        sample_size=len(features),
        shap_values=values,
        sample=features,
    )


def explain_prediction(artifact: ModelArtifact, row: pd.DataFrame) -> PredictionExplanation:
    """Explain a single prediction.

    Args:
        row: A one-row frame. A Series is not accepted, because the pipeline
            expects a frame and silently mis-shaping it is the sort of thing
            that produces a confident wrong answer.

    Raises:
        ValueError: if ``row`` is not exactly one row.
    """
    if len(row) != 1:
        raise ValueError(f"explain_prediction expects exactly one row, got {len(row)}")

    ordered = row[list(artifact.feature_columns)]
    features = _preprocess(artifact.pipeline, ordered)
    explanation = _tree_explainer(artifact.pipeline)(features)

    values = np.asarray(explanation.values).reshape(-1)
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    prediction = float(artifact.pipeline.predict(ordered)[0])

    contributions = [
        Contribution(
            feature=str(name),
            value=features.iloc[0][name],
            shap_value=float(value),
            effect_multiplier=float(np.exp(value)),
        )
        for name, value in zip(features.columns, values, strict=True)
        # A feature contributing nothing is noise in an explanation meant to
        # be read by a person.
        if value != 0.0
    ]

    return PredictionExplanation(
        prediction_eur=prediction,
        # The base value is a log-space prediction, so expm1 maps it to EUR
        # the same way the model's own inverse transform does.
        base_value_eur=float(np.expm1(base)),
        contributions=contributions,
    )
