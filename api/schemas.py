"""Request and response schemas — the frozen surface of the API.

These classes *are* the contract in docs/API_CONTRACT.md. Every field carries a
description and the models carry examples, because an OpenAPI document with
bare types is a type checker rather than documentation, and Phase 10's consumer
should not have to read this file to know what a field means.

Pydantic v2 idioms only: ``model_config = ConfigDict(...)``, ``@model_validator``,
``.model_dump()``. The v1 spellings still run and warn, and the suite turns
DeprecationWarning into an error.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.prediction import DEFAULT_TOP_N, MAX_TOP_N

Variant = Literal["performance_only", "with_prior_value"]


class ErrorBody(BaseModel):
    """The inner half of every error response."""

    code: str = Field(description="Stable machine-readable code. Branch on this.")
    message: str = Field(description="Human-readable explanation. May be reworded.")
    detail: Any = Field(default=None, description="Field-level errors, when there are any.")


class ErrorResponse(BaseModel):
    """One envelope for every non-2xx response, so a client writes one error path."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "player_not_found",
                    "message": "no player with id 999999999",
                    "detail": None,
                }
            }
        }
    )

    error: ErrorBody


class HealthResponse(BaseModel):
    """Liveness and readiness. Deliberately unversioned."""

    status: Literal["ok", "degraded"] = Field(
        description="'degraded' means the process is up but cannot predict."
    )
    ready: bool = Field(description="Whether a model artifact is loaded.")
    models_loaded: list[str] = Field(description="Variants available for prediction.")
    version: str = Field(description="Application version.")


class PredictRequest(BaseModel):
    """Exactly one of ``player_id`` or ``features``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"player_id": 28003, "variant": "performance_only"},
                {
                    "features": {
                        "age": 24.5,
                        "goals": 12,
                        "minutes_played": 2400,
                        "position": "Attack",
                    },
                    "variant": "performance_only",
                },
            ]
        },
    )

    player_id: int | None = Field(
        default=None, ge=1, description="Predict from this player's stored season."
    )
    season: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Season to use. Omitted, the most recent on record is used.",
    )
    features: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Explicit feature values. Unknown keys are rejected; omitted keys "
            "are imputed exactly as during training."
        ),
    )
    variant: Variant | None = Field(
        default=None,
        description=(
            "Which model to use. 'performance_only' answers scouting questions; "
            "'with_prior_value' is more accurate but anchored to a known valuation."
        ),
    )
    top_n: Annotated[int, Field(ge=1, le=MAX_TOP_N)] = Field(
        default=DEFAULT_TOP_N, description="Contributions returned per direction."
    )

    @model_validator(mode="after")
    def _exactly_one_input(self) -> Self:
        supplied = [self.player_id is not None, self.features is not None]
        if sum(supplied) != 1:
            raise ValueError(
                "supply exactly one of 'player_id' or 'features', not both and not neither"
            )
        if self.season is not None and self.player_id is None:
            raise ValueError("'season' only applies together with 'player_id'")
        return self


class ContributionSchema(BaseModel):
    """One feature's push on one prediction, in log space."""

    feature: str
    value: Any = Field(description="The feature's value for this row.")
    shap_value: float = Field(
        description=(
            "Additive contribution in log space. NOT additive in euros — never "
            "sum these into a euro figure."
        )
    )
    effect_multiplier: float = Field(
        description=(
            "exp(shap_value): what this feature multiplied the prediction by. "
            "Exact at any value; 1.51 means it raised the prediction by 51%."
        )
    )
    direction: Literal["increases", "decreases"]


class ConfidenceSchema(BaseModel):
    """An empirical prediction interval. Not a probability.

    A gradient booster has no calibrated uncertainty, so this reports the
    model's own measured residual quantiles on held-out seasons for the value
    band the prediction falls into.

    Two coverage numbers, deliberately. ``level`` is what the interval asks
    for; ``measured_coverage`` is what it got on seasons neither the model nor
    the interval had seen. They differ, and a response that carried only the
    first would be quoting its own intention as a result.
    """

    level: float = Field(
        description=(
            "Nominal coverage the interval targets. 0.8 asks for 80%; "
            "`measured_coverage` reports what it achieved."
        )
    )
    lower_eur: float
    upper_eur: float
    basis: str = Field(description="What the interval was measured from.")
    reference_rows: int = Field(description="Rows it was measured over. Fewer rows, less trust.")
    measured_coverage: float | None = Field(
        default=None,
        description=(
            "Fraction of test-season rows the interval actually contained, measured on "
            "seasons after the one its quantiles came from. Below `level` because "
            "consecutive seasons are not exchangeable. Null if never measured."
        ),
    )


class ModelSummary(BaseModel):
    """Enough to identify which model produced a prediction."""

    name: str
    variant: str
    trained_at: str


class ExplanationSchema(BaseModel):
    """Why the model landed where it did."""

    base_value_eur: float = Field(
        description="What the model predicts before any feature is considered."
    )
    top_positive_features: list[ContributionSchema] = Field(
        description="Largest upward contributions, biggest first."
    )
    top_negative_features: list[ContributionSchema] = Field(
        description="Largest downward contributions, biggest first."
    )


class PredictResponse(BaseModel):
    """The prediction, and everything needed to judge it."""

    prediction_eur: float = Field(description="Predicted market value, in EUR.")
    variant: str
    model: ModelSummary
    player_id: int | None = None
    season: int | None = Field(
        default=None, description="The season actually used, so the input period is auditable."
    )
    confidence: ConfidenceSchema | None = Field(
        default=None, description="Absent when the model carries no measured calibration."
    )
    explanation: ExplanationSchema | None = Field(
        default=None, description="Absent for model families SHAP cannot explain."
    )


class SeasonRow(BaseModel):
    """One player-season on record."""

    season: int
    age: float
    appearances: int
    goals: int
    assists: int
    minutes_played: int
    market_value_in_eur: float | None = Field(
        default=None,
        description=(
            "Recorded market value, or null for a season still being played. "
            "Null means 'not yet published', never zero."
        ),
    )
    has_label: bool = Field(
        default=True,
        description=(
            "False for the season in progress: its statistics are complete "
            "enough to predict from, but no valuation has been published to "
            "compare against."
        ),
    )


class PlayerSearchResult(BaseModel):
    """One hit from a name search."""

    player_id: int
    name: str
    club: str | None = Field(
        default=None,
        description="The club he is at now — not the club of the season being predicted.",
    )
    league: str | None = Field(
        default=None, description="His club's domestic competition, e.g. 'Premier League'."
    )
    league_country: str | None = Field(
        default=None,
        description=(
            "The country that competition belongs to. Sent because the names do not "
            "identify themselves: two leagues here are called 'Premier Liga'."
        ),
    )
    position: str | None = None
    latest_season: int | None = None
    market_value_in_eur: float | None = None
    predictable: bool = Field(
        description="Whether this player has a modellable season. A result that leads to a 404 is worse than no result."
    )


class PlayerSearchResponse(BaseModel):
    query: str
    results: list[PlayerSearchResult]


class SimilarPlayer(BaseModel):
    """A comparable season, measured in the model's own feature space."""

    player_id: int
    name: str | None = None
    season: int
    position: str | None = None
    age: float
    market_value_in_eur: float
    distance: float = Field(
        description="Euclidean distance on preprocessed features. Smaller is more alike."
    )


class SimilarPlayersResponse(BaseModel):
    player_id: int
    season: int | None = None
    results: list[SimilarPlayer]


class HistoryPoint(BaseModel):
    """One season, predicted against what actually happened."""

    season: int
    predicted_eur: float
    actual_eur: float
    error_eur: float = Field(description="predicted − actual, so positive means overvalued")
    in_training_range: bool = Field(
        description="The model trained on this season. Agreement here is not evidence."
    )
    held_out: bool = Field(description="A test season the model never saw.")


class PredictionHistoryResponse(BaseModel):
    player_id: int
    variant: str
    points: list[HistoryPoint]


class FeatureDistributionResponse(BaseModel):
    """Population quantiles, so a client can place a value in context."""

    variant: str
    grid: list[float] = Field(description="Quantile levels the values correspond to.")
    features: dict[str, Any] = Field(
        description="Per feature: quantiles at each grid level, median and n."
    )


class PlayerResponse(BaseModel):
    """A player's attributes and every season on file."""

    player_id: int
    name: str | None = None
    club: str | None = Field(
        default=None,
        description="The club he is at now — not the club of the season being predicted.",
    )
    league: str | None = Field(
        default=None, description="His club's domestic competition, e.g. 'Premier League'."
    )
    league_country: str | None = Field(
        default=None,
        description=(
            "The country that competition belongs to. Sent because the names do not "
            "identify themselves: two leagues here are called 'Premier Liga'."
        ),
    )
    position: str | None = None
    sub_position: str | None = None
    foot: str | None = None
    height_in_cm: float | None = None
    country_of_citizenship: str | None = None
    seasons: list[SeasonRow]
    features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Model-ready feature values for the latest season. Seeds a what-if "
            "form with the real starting point instead of a guess."
        ),
    )


class MetricsSchema(BaseModel):
    """Metrics in EUR, never in log space."""

    mae_eur: float | None = None
    rmse_eur: float | None = None
    r2: float | None = None
    mape: float | None = None
    n: int | None = None


class ModelMetricsResponse(BaseModel):
    variant: str
    validation: MetricsSchema
    test: MetricsSchema = Field(
        description="Held-out seasons the model never saw. The reported headline."
    )
    leaderboard: list[dict[str, Any]] = Field(
        default_factory=list, description="Every family tried, ranked by validation MAE."
    )


class ModelInfoResponse(BaseModel):
    variant: str
    model_name: str
    params: dict[str, Any]
    feature_columns: list[str]
    target_column: str
    trained_at: str
    seed: int
    split: dict[str, Any]
    dataset: dict[str, Any]
    artifact_version: int
    explainable: bool = Field(description="Whether SHAP explanations are available.")


class ModelListResponse(BaseModel):
    variants: list[str]
    default: str | None = Field(default=None, description="Used when a request omits 'variant'.")


class FeatureImportanceEntry(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    """The estimator's own importances, plus global SHAP where available."""

    variant: str
    model_name: str
    features: list[FeatureImportanceEntry]
    shap: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Global SHAP impact. Importances say what the model used; SHAP says "
            "how much each feature moved predictions."
        ),
    )
