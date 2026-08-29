"""The prediction path, with no web framework anywhere in it.

Nothing in this module imports FastAPI, Starlette or Pydantic. That is the
whole point of it existing: the same code serves an HTTP request, a batch job,
a CLI and a test, and the tests for prediction logic need no running server.
Where something goes wrong the service raises its own exceptions, which the
transport layer maps to status codes — a service that raised ``HTTPException``
would be a web handler wearing a different name.

Layering, top to bottom:

    router          HTTP: schemas, status codes
    PredictionService   <- here
    artifact        fitted preprocessing + model, as one object
    explainer       SHAP contributions as data
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.explainability.shap_explainer import (
    PredictionExplanation,
    explain_global,
    explain_prediction,
    supports_shap,
)
from src.feature_engineering.build import TARGET_COLUMN
from src.models.artifact import ModelArtifact, load, predict
from src.models.calibration import interval_for
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_N = 5
MAX_TOP_N = 25
"""How many contributions a response carries. The service always computes the
full explanation; truncation belongs to the caller, which is why these are
defaults for the transport layer rather than parameters of the service."""


class ServiceError(Exception):
    """Base for every failure this service reports.

    Carries a stable ``code`` because that is what a client branches on; the
    message is for a human reading a log and may be reworded at any time.
    """

    code = "service_error"

    def __init__(self, message: str, detail: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class PlayerNotFoundError(ServiceError):
    code = "player_not_found"


class SeasonNotFoundError(ServiceError):
    code = "season_not_found"


class ModelNotFoundError(ServiceError):
    code = "model_not_found"


class ModelUnavailableError(ServiceError):
    code = "model_unavailable"


class InvalidFeaturesError(ServiceError):
    code = "validation_error"


@dataclass(frozen=True)
class PredictionResult:
    """Everything one prediction produced, before any HTTP shaping."""

    prediction_eur: float
    variant: str
    model_name: str
    trained_at: str
    player_id: int | None
    season: int | None
    confidence: dict[str, Any]
    explanation: PredictionExplanation | None

    def top_positive(self, n: int) -> list[dict[str, Any]]:
        if self.explanation is None:
            return []
        return [c.as_dict() for c in self.explanation.positive()[:n]]

    def top_negative(self, n: int) -> list[dict[str, Any]]:
        if self.explanation is None:
            return []
        return [c.as_dict() for c in self.explanation.negative()[:n]]


class PredictionService:
    """Loaded models and player history, behind one object.

    Constructed once per process. Loading is eager and explicit rather than
    lazy-on-first-request, so a broken artifact fails at startup where someone
    is watching, instead of on a user's first prediction.
    """

    def __init__(
        self, artifacts: Mapping[str, ModelArtifact], players: pd.DataFrame | None = None
    ) -> None:
        self._artifacts = dict(artifacts)
        self._players = players if players is not None else pd.DataFrame()
        self._explainable = {
            name: supports_shap(artifact) for name, artifact in self._artifacts.items()
        }

    # -- construction ----------------------------------------------------

    @classmethod
    def from_directory(
        cls, model_directory: Path, players: pd.DataFrame | None = None
    ) -> PredictionService:
        """Load every artifact in a directory, newest per variant."""
        artifacts: dict[str, ModelArtifact] = {}
        for path in sorted(model_directory.glob("*.joblib")):
            try:
                artifact = load(path)
            except ValueError as exc:
                # A schema mismatch is a real problem, but one stale file must
                # not stop the service loading the artifacts that are fine.
                logger.warning("skipping %s: %s", path.name, exc)
                continue
            artifacts[artifact.variant] = artifact
            logger.info("loaded %s (%s)", artifact.variant, artifact.model_name)

        if not artifacts:
            logger.warning("no usable model artifacts in %s", model_directory)
        return cls(artifacts, players)

    # -- introspection ---------------------------------------------------

    @property
    def variants(self) -> list[str]:
        return sorted(self._artifacts)

    @property
    def ready(self) -> bool:
        """Whether the service can actually answer a prediction request."""
        return bool(self._artifacts)

    def artifact(self, variant: str | None = None) -> ModelArtifact:
        """The artifact for ``variant``, or the default one.

        Raises:
            ModelUnavailableError: if nothing is loaded.
            ModelNotFoundError: if that variant is not among the loaded models.
        """
        if not self._artifacts:
            raise ModelUnavailableError("no model artifact is loaded; run scripts/train_models.py")
        if variant is None:
            return self._artifacts[self.variants[0]]
        if variant not in self._artifacts:
            raise ModelNotFoundError(
                f"no model for variant {variant!r}; loaded: {', '.join(self.variants)}"
            )
        return self._artifacts[variant]

    def model_info(self, variant: str | None = None) -> dict[str, Any]:
        artifact = self.artifact(variant)
        return {
            "variant": artifact.variant,
            "model_name": artifact.model_name,
            "params": artifact.params,
            "feature_columns": list(artifact.feature_columns),
            "target_column": artifact.target_column,
            "trained_at": artifact.created_at,
            "seed": artifact.seed,
            "split": artifact.split,
            "dataset": artifact.dataset,
            "artifact_version": artifact.artifact_version,
            "explainable": self._explainable.get(artifact.variant, False),
        }

    def metrics(self, variant: str | None = None) -> dict[str, Any]:
        artifact = self.artifact(variant)
        return {
            "variant": artifact.variant,
            "validation": _metrics(artifact.validation),
            "test": _metrics(artifact.test),
            "leaderboard": artifact.leaderboard,
        }

    def feature_importance(self, variant: str | None = None, *, top_n: int = 20) -> dict[str, Any]:
        artifact = self.artifact(variant)
        ranked = sorted(artifact.feature_importance.items(), key=lambda kv: -abs(kv[1]))
        return {
            "variant": artifact.variant,
            "model_name": artifact.model_name,
            "features": [{"feature": name, "importance": value} for name, value in ranked[:top_n]],
        }

    def global_explanation(
        self, variant: str | None = None, *, sample_size: int = 500, top_n: int = 20
    ) -> dict[str, Any]:
        """SHAP over stored player rows. Empty when there is nothing to sample."""
        artifact = self.artifact(variant)
        if self._players.empty or not self._explainable.get(artifact.variant, False):
            return {}

        rows = self._rows_for_variant(artifact)
        if rows.empty:
            return {}
        return explain_global(artifact, rows, sample_size=sample_size).as_dict(top_n)

    # -- player lookup ---------------------------------------------------

    def player(self, player_id: int) -> dict[str, Any]:
        """A player's record and every season on file.

        Raises:
            PlayerNotFoundError: if the id is not in the dataset.
        """
        rows = self._player_rows(player_id)
        latest = rows.iloc[-1]

        return {
            "player_id": int(player_id),
            "position": _plain(latest.get("position")),
            "sub_position": _plain(latest.get("sub_position")),
            "foot": _plain(latest.get("foot")),
            "height_in_cm": _plain(latest.get("height_in_cm")),
            "country_of_citizenship": _plain(latest.get("country_of_citizenship")),
            "seasons": [
                {
                    "season": int(row["season"]),
                    "age": float(row["age"]),
                    "appearances": int(row["appearances"]),
                    "goals": int(row["goals"]),
                    "assists": int(row["assists"]),
                    "minutes_played": int(row["minutes_played"]),
                    "market_value_in_eur": float(row[TARGET_COLUMN]),
                }
                for _, row in rows.iterrows()
            ],
        }

    def _player_rows(self, player_id: int) -> pd.DataFrame:
        if self._players.empty:
            raise PlayerNotFoundError(f"no player data is loaded (asked for {player_id})")

        rows = self._players[self._players["player_id"] == player_id]
        if rows.empty:
            raise PlayerNotFoundError(f"no player with id {player_id}")
        return rows.sort_values("season")

    def _rows_for_variant(self, artifact: ModelArtifact) -> pd.DataFrame:
        """Rows usable by this variant.

        The prior-value model cannot predict for a player's first season, so
        those rows are excluded rather than imputed into a confident guess.
        """
        missing = [c for c in artifact.feature_columns if c not in self._players.columns]
        if missing:
            return pd.DataFrame()
        return self._players.dropna(subset=list(artifact.feature_columns))

    # -- prediction ------------------------------------------------------

    def predict_for_player(
        self,
        player_id: int,
        *,
        season: int | None = None,
        variant: str | None = None,
    ) -> PredictionResult:
        """Predict using a stored player-season.

        Raises:
            PlayerNotFoundError, SeasonNotFoundError, ModelNotFoundError, ModelUnavailableError.
        """
        artifact = self.artifact(variant)
        rows = self._player_rows(player_id)

        if season is not None:
            rows = rows[rows["season"] == season]
            if rows.empty:
                raise SeasonNotFoundError(f"player {player_id} has no row for season {season}")

        row = rows.iloc[[-1]]
        missing = [
            column
            for column in artifact.feature_columns
            if column not in row.columns or row.iloc[0][column] is None
        ]
        if missing:
            raise InvalidFeaturesError(
                f"stored row for player {player_id} lacks {', '.join(missing)}",
                detail=missing,
            )

        return self._predict(
            artifact,
            row,
            player_id=int(player_id),
            season=int(row.iloc[0]["season"]),
        )

    def predict_from_features(
        self,
        features: Mapping[str, Any],
        *,
        variant: str | None = None,
    ) -> PredictionResult:
        """Predict from an explicit feature mapping.

        Unknown keys are rejected rather than dropped: silently ignoring a
        misspelt ``minutes_playd`` returns a confident answer to a question
        nobody asked. Omitted keys are imputed by the fitted pipeline, exactly
        as they were during training.

        Raises:
            InvalidFeaturesError: on an unknown key.
        """
        artifact = self.artifact(variant)
        expected = set(artifact.feature_columns)

        unknown = sorted(set(features) - expected)
        if unknown:
            raise InvalidFeaturesError(f"unknown feature(s): {', '.join(unknown)}", detail=unknown)

        row = pd.DataFrame([{column: features.get(column) for column in artifact.feature_columns}])
        return self._predict(artifact, row, player_id=None, season=None)

    def _predict(
        self,
        artifact: ModelArtifact,
        row: pd.DataFrame,
        *,
        player_id: int | None,
        season: int | None,
    ) -> PredictionResult:
        prediction = float(predict(artifact, row)[0])

        explanation = None
        if self._explainable.get(artifact.variant, False):
            explanation = explain_prediction(artifact, row)

        return PredictionResult(
            prediction_eur=prediction,
            variant=artifact.variant,
            model_name=artifact.model_name,
            trained_at=artifact.created_at,
            player_id=player_id,
            season=season,
            confidence=interval_for(artifact.calibration, prediction),
            explanation=explanation,
        )


def _metrics(metrics: object) -> dict[str, Any]:
    return {
        "mae_eur": getattr(metrics, "mae", None),
        "rmse_eur": getattr(metrics, "rmse", None),
        "r2": getattr(metrics, "r2", None),
        "mape": getattr(metrics, "mape", None),
        "n": getattr(metrics, "n", None),
    }


def _plain(value: object) -> object:
    if value is None or (isinstance(value, float) and value != value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else value
