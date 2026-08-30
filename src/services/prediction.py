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

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.explainability.shap_explainer import (
    PredictionExplanation,
    explain_global,
    explain_prediction,
    supports_shap,
)
from src.feature_engineering.build import NON_NEGATIVE_FEATURES, TARGET_COLUMN
from src.models.artifact import ModelArtifact, load, predict
from src.models.calibration import interval_for
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_N = 5
MAX_TOP_N = 25
MAX_SEARCH_RESULTS = 50
DEFAULT_SIMILAR = 8

QUANTILE_GRID = tuple(round(0.05 * step, 2) for step in range(21))
"""0, 0.05 ... 1.0. A grid rather than a handful of named quantiles so a
client can interpolate a value's percentile without another request — which is
what a radar chart needs to place an axis honestly."""

MAX_CATEGORY_LENGTH = 200
"""Longest accepted categorical value. Every real category in this dataset is
a position, a foot or a country name; nothing legitimate approaches this. The
cap exists because ``/api/v1/predict`` is unauthenticated, and an unbounded
string is free memory and free log volume for anyone who asks."""


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

    Constructed once per process, in the lifespan handler — eagerly rather than
    on first request, so the cost of reading a hundred megabytes of artifact is
    paid at startup where someone is watching, and `/health` can answer
    truthfully from the moment it is up.

    Eager does not mean fatal. :meth:`from_directory` logs and skips an
    artifact it cannot read rather than refusing to start: one stale file must
    not take down the variants that are fine, and `/health` reports `ready`
    from what actually loaded. A process that is alive with nothing to serve
    still answers 200 with ``ready: false`` — the honest state, and the one an
    orchestrator can act on without restart-looping.
    """

    def __init__(
        self,
        artifacts: Mapping[str, ModelArtifact],
        players: pd.DataFrame | None = None,
        names: Mapping[int, str] | None = None,
    ) -> None:
        self._artifacts = dict(artifacts)
        self._players = players if players is not None else pd.DataFrame()
        self._names = dict(names or {})
        self._explainable = {
            name: supports_shap(artifact) for name, artifact in self._artifacts.items()
        }
        # Built lazily: fitting the neighbour index costs a pass over every row
        # and most requests never ask for similar players.
        self._neighbours: dict[str, tuple[NearestNeighbors, pd.DataFrame]] = {}
        self._distribution: dict[str, Any] = {}
        self._numeric_columns: dict[str, frozenset[str]] = {}

    def name_for(self, player_id: int) -> str | None:
        """Display name, when the players table was loaded alongside."""
        return self._names.get(int(player_id))

    # -- construction ----------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        model_directory: Path,
        players: pd.DataFrame | None = None,
        names: Mapping[int, str] | None = None,
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
        return cls(artifacts, players, names)

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
            "name": self.name_for(player_id),
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
            "features": self.features_for(player_id),
        }

    def features_for(
        self, player_id: int, *, season: int | None = None, variant: str | None = None
    ) -> dict[str, Any]:
        """The model-ready feature values for one player-season.

        Returned so a client can seed a what-if form with the real starting
        point rather than reconstructing it, and so "change goals from 10 to
        18" is a change from something true.
        """
        artifact = self.artifact(variant)
        rows = self._player_rows(player_id)
        if season is not None:
            rows = rows[rows["season"] == season]
            if rows.empty:
                raise SeasonNotFoundError(f"player {player_id} has no row for season {season}")

        latest = rows.iloc[-1]
        return {
            column: _plain(latest.get(column))
            for column in artifact.feature_columns
            if column in rows.columns
        }

    # -- search and neighbours -------------------------------------------

    def search_players(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Find players by name, case-insensitively.

        Substring rather than fuzzy matching. Fuzzy search needs a measured
        threshold and a way to judge a bad match, and neither exists yet; a
        substring match is honest about being exactly what it is.
        """
        cleaned = query.strip().lower()
        if not cleaned:
            return []
        if not self._names:
            raise PlayerNotFoundError("no player names are loaded; search is unavailable")

        modelled = set(self._players["player_id"]) if not self._players.empty else set()
        matches = [
            (player_id, name) for player_id, name in self._names.items() if cleaned in name.lower()
        ]
        # Players the model can actually predict for come first: a search
        # result that leads to a 404 is worse than no result.
        matches.sort(key=lambda pair: (pair[0] not in modelled, len(pair[1]), pair[1]))

        results = []
        for player_id, name in matches[: min(limit, MAX_SEARCH_RESULTS)]:
            row = self._latest_row(player_id)
            results.append(
                {
                    "player_id": int(player_id),
                    "name": name,
                    "position": _plain(row.get("position")) if row is not None else None,
                    "latest_season": int(row["season"]) if row is not None else None,
                    "market_value_in_eur": (float(row[TARGET_COLUMN]) if row is not None else None),
                    "predictable": player_id in modelled,
                }
            )
        return results

    def _latest_row(self, player_id: int) -> pd.Series | None:
        if self._players.empty:
            return None
        rows = self._players[self._players["player_id"] == player_id]
        if rows.empty:
            return None
        return rows.sort_values("season").iloc[-1]

    def similar_players(
        self, player_id: int, *, k: int = DEFAULT_SIMILAR, variant: str | None = None
    ) -> list[dict[str, Any]]:
        """Players whose season most resembles this one, in the model's own space.

        Distance is measured on the *preprocessed* features — the same scaled,
        encoded matrix the model sees — so "similar" means similar to the model
        rather than similar on a hand-picked pair of columns.

        Restricted to the same season, because market conditions differ across
        years and a 2014 striker is not a comparison for a 2024 one.
        """
        artifact = self.artifact(variant)
        rows = self._player_rows(player_id)
        season = int(rows.iloc[-1]["season"])

        index, pool = self._neighbour_index(artifact, season)
        if index is None or len(pool) <= 1:
            return []

        query = pool[pool["player_id"] == player_id]
        if query.empty:
            return []

        features = artifact.pipeline.named_steps["preprocess"].transform(
            query[list(artifact.feature_columns)]
        )
        wanted = min(k + 1, len(pool))
        distances, positions = index.kneighbors(features, n_neighbors=wanted)

        results = []
        for distance, position in zip(distances[0], positions[0], strict=True):
            row = pool.iloc[position]
            if int(row["player_id"]) == int(player_id):
                continue
            results.append(
                {
                    "player_id": int(row["player_id"]),
                    "name": self.name_for(int(row["player_id"])),
                    "season": int(row["season"]),
                    "position": _plain(row.get("position")),
                    "age": float(row["age"]),
                    "market_value_in_eur": float(row[TARGET_COLUMN]),
                    "distance": float(distance),
                }
            )
        return results[:k]

    def _neighbour_index(
        self, artifact: ModelArtifact, season: int
    ) -> tuple[NearestNeighbors | None, pd.DataFrame]:
        cache_key = f"{artifact.variant}:{season}"
        if cache_key in self._neighbours:
            index, pool = self._neighbours[cache_key]
            return index, pool

        pool = self._rows_for_variant(artifact)
        pool = pool[pool["season"] == season] if not pool.empty else pool
        if pool.empty:
            return None, pool

        matrix = artifact.pipeline.named_steps["preprocess"].transform(
            pool[list(artifact.feature_columns)]
        )
        # Fitted on the frame, not a bare array: the preprocessor emits named
        # columns, and querying a name-less fit with a named frame makes
        # sklearn warn that the two do not agree. They should agree.
        index = NearestNeighbors(n_neighbors=min(DEFAULT_SIMILAR + 1, len(pool)))
        index.fit(matrix)

        self._neighbours[cache_key] = (index, pool)
        return index, pool

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

    def feature_distribution(self, variant: str | None = None) -> dict[str, Any]:
        """Quantiles per numeric feature, across the whole panel.

        A radar axis needs to know what "good" looks like. Normalising against
        only the two players being compared would peg one of them at the
        maximum on every axis and say nothing; normalising against the
        population says "92nd percentile for goals per 90", which is a fact.
        """
        artifact = self.artifact(variant)
        if artifact.variant in self._distribution:
            cached: dict[str, Any] = self._distribution[artifact.variant]
            return cached
        if self._players.empty:
            return {}

        features: dict[str, Any] = {}
        for column in artifact.feature_columns:
            if column not in self._players.columns:
                continue
            series = pd.to_numeric(self._players[column], errors="coerce").dropna()
            if series.empty:
                continue
            features[column] = {
                "quantiles": [float(series.quantile(q)) for q in QUANTILE_GRID],
                "median": float(series.median()),
                "n": int(len(series)),
            }

        payload = {
            "variant": artifact.variant,
            "grid": list(QUANTILE_GRID),
            "features": features,
        }
        self._distribution[artifact.variant] = payload
        return payload

    def prediction_history(
        self, player_id: int, *, variant: str | None = None
    ) -> list[dict[str, Any]]:
        """Predict every season on record, beside what actually happened.

        One prediction says what the model thinks. A series says whether it
        tracks a career or merely lands near the middle — and the seasons in
        the training range will look better than the held-out ones, which is
        itself worth seeing.
        """
        artifact = self.artifact(variant)
        rows = self._player_rows(player_id)

        usable = rows.dropna(subset=list(artifact.feature_columns))
        if usable.empty:
            return []

        predictions = predict(artifact, usable)
        split = artifact.split or {}
        test_start = split.get("test_start_season")
        train_end = split.get("train_end_season")

        history = []
        for (_, row), predicted in zip(usable.iterrows(), predictions, strict=True):
            season = int(row["season"])
            actual = float(row[TARGET_COLUMN])
            history.append(
                {
                    "season": season,
                    "predicted_eur": float(predicted),
                    "actual_eur": actual,
                    "error_eur": float(predicted) - actual,
                    # Stated per row, because a prediction for a season the
                    # model trained on is not evidence of anything.
                    "in_training_range": (train_end is not None and season <= int(train_end)),
                    "held_out": (test_start is not None and season >= int(test_start)),
                }
            )
        return history

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
            InvalidFeaturesError: on an unknown key, or a value the model
                cannot be asked about — a wrong type, a non-finite number, a
                negative count, or an implausibly long category.
        """
        artifact = self.artifact(variant)
        expected = set(artifact.feature_columns)

        unknown = sorted(set(features) - expected)
        if unknown:
            raise InvalidFeaturesError(f"unknown feature(s): {', '.join(unknown)}", detail=unknown)

        self._validate_values(artifact, features)

        row = pd.DataFrame([{column: features.get(column) for column in artifact.feature_columns}])
        try:
            return self._predict(artifact, row, player_id=None, season=None)
        except (ValueError, TypeError) as exc:
            # Backstop. _validate_values catches every malformed value seen so
            # far, but the fitted pipeline is the only thing that knows its own
            # full expectations, and a value it rejects is a bad request rather
            # than a broken server. Without this the caller gets a bare 500 for
            # something they typed.
            raise InvalidFeaturesError(
                f"the model could not be evaluated on these features: {exc}",
                detail=sorted(features),
            ) from exc

    def _numeric_features(self, artifact: ModelArtifact) -> frozenset[str]:
        """Which of this artifact's features the preprocessor treats as numeric.

        Read from the fitted ColumnTransformer rather than re-listed here, so
        it cannot disagree with what the model was actually trained on.
        """
        cached = self._numeric_columns.get(artifact.variant)
        if cached is not None:
            return cached

        numeric: set[str] = set()
        try:
            columns = artifact.pipeline.named_steps["preprocess"].named_steps["columns"]
            for name, _, selected in columns.transformers_:
                if name == "numeric":
                    numeric.update(str(column) for column in selected)
        except (AttributeError, KeyError):  # pragma: no cover - defensive
            # An artifact built by another pipeline shape still predicts; it
            # just falls back to the type and finiteness checks below.
            pass

        resolved = frozenset(numeric)
        self._numeric_columns[artifact.variant] = resolved
        return resolved

    def _validate_values(self, artifact: ModelArtifact, features: Mapping[str, Any]) -> None:
        """Reject values the model must not be asked to answer for.

        The keys were checked above; these are the values. Rejecting them here
        rather than letting scikit-learn raise deep inside a transformer is
        what turns an opaque 500 into a 422 that names the offending feature.

        Raises:
            InvalidFeaturesError: listing every bad feature, not just the first.
        """
        numeric = self._numeric_features(artifact)
        problems: list[str] = []

        for name, value in features.items():
            if value is None:
                # Explicitly absent. The fitted imputer fills it exactly as it
                # did during training, which is the documented behaviour.
                continue

            if isinstance(value, (dict, list, tuple, set)):
                problems.append(f"{name}: expected a single value, got {type(value).__name__}")
                continue

            if name in numeric:
                if isinstance(value, bool):
                    # float(True) is 1.0, so this would otherwise be accepted
                    # silently — a confident answer to a question nobody asked.
                    problems.append(f"{name}: expected a number, got a boolean")
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    problems.append(f"{name}: expected a number, got {value!r}")
                    continue
                if not math.isfinite(number):
                    problems.append(f"{name}: expected a finite number, got {value!r}")
                elif name in NON_NEGATIVE_FEATURES and number < 0:
                    problems.append(f"{name}: cannot be negative, got {number:g}")
            elif isinstance(value, str) and len(value) > MAX_CATEGORY_LENGTH:
                problems.append(f"{name}: category longer than {MAX_CATEGORY_LENGTH} characters")

        if problems:
            raise InvalidFeaturesError(
                f"invalid feature value(s): {'; '.join(problems)}", detail=problems
            )

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
