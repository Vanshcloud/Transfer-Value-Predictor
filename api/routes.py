"""HTTP transport. Schemas in, service calls out, status codes on the way back.

Every handler here is a few lines: validate, call the service, shape the
response. Anything longer than that belongs in
:mod:`src.services.prediction`, which knows nothing about HTTP and can
therefore be reused by a batch job or a CLI.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query

from api.dependencies import ServiceDep
from api.limits import inference_slot
from api.schemas import (
    ConfidenceSchema,
    ErrorResponse,
    ExplanationSchema,
    FeatureDistributionResponse,
    FeatureImportanceResponse,
    ModelInfoResponse,
    ModelListResponse,
    ModelMetricsResponse,
    ModelSummary,
    PlayerResponse,
    PlayerSearchResponse,
    PredictionHistoryResponse,
    PredictRequest,
    PredictResponse,
    SimilarPlayersResponse,
)

router = APIRouter(prefix="/api/v1")

VariantPath = Annotated[str, Path(description="performance_only or with_prior_value")]


def errors(declared: dict[int, str], *, validated: bool = True) -> dict[int | str, Any]:
    """The error responses a route can send, every one in the `ErrorResponse` envelope.

    Without a model, FastAPI documents a 422 as its own `HTTPValidationError`
    (`{"detail": [...]}`) and the rest with no body at all, while every one
    of them is actually sent as `{"error": {...}}` by `api.errors`. A client
    generated from the old document parsed a shape it never received.
    `validated` adds the 422 for a route with parameters — FastAPI documents
    one on every such route whether or not a parameter can fail, so it is
    declared here in the shape actually sent. 500 is always possible. `tests/unit/test_api.py::TestErrorResponsesAreDocumented` provokes each one.
    """
    statuses = {
        **declared,
        **({422: "A parameter or the body failed validation"} if validated else {}),
        500: "Unhandled server error; the message is fixed and the detail is null",
    }
    return {
        status: {"model": ErrorResponse, "description": description}
        for status, description in statuses.items()
    }


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict a market value, with an explanation",
    responses=errors(
        {
            404: "Player, season or model not found",
            413: "Request body larger than 64 KB",
            422: "Malformed body, an unknown feature name or an invalid feature value",
            503: "No model artifact is loaded, or every inference slot is busy",
        }
    ),
    dependencies=[Depends(inference_slot)],
)
def predict(request: PredictRequest, service: ServiceDep) -> PredictResponse:
    """Predict from a stored player-season, or from explicit features.

    Exactly one of `player_id` or `features` must be supplied; the schema
    rejects both and neither.
    """
    if request.player_id is not None:
        result = service.predict_for_player(
            request.player_id, season=request.season, variant=request.variant
        )
    else:
        result = service.predict_from_features(request.features or {}, variant=request.variant)

    explanation = None
    if result.explanation is not None:
        explanation = ExplanationSchema(
            base_value_eur=result.explanation.base_value_eur,
            top_positive_features=result.top_positive(request.top_n),  # type: ignore[arg-type]
            top_negative_features=result.top_negative(request.top_n),  # type: ignore[arg-type]
        )

    return PredictResponse(
        prediction_eur=result.prediction_eur,
        variant=result.variant,
        model=ModelSummary(
            name=result.model_name, variant=result.variant, trained_at=result.trained_at
        ),
        player_id=result.player_id,
        season=result.season,
        confidence=ConfidenceSchema(**result.confidence) if result.confidence else None,
        explanation=explanation,
    )


@router.get(
    "/players",
    response_model=PlayerSearchResponse,
    summary="Find players by name",
    responses=errors({404: "No player names are loaded"}),
)
def search_players(
    service: ServiceDep,
    q: Annotated[str, Query(min_length=1, max_length=100, description="Name substring")],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> PlayerSearchResponse:
    """Case-insensitive substring match, modellable players first."""
    return PlayerSearchResponse(query=q, results=service.search_players(q, limit=limit))  # type: ignore[arg-type]


@router.get(
    "/players/{player_id}/similar",
    response_model=SimilarPlayersResponse,
    summary="Comparable seasons, in the model's own feature space",
    responses=errors({404: "No such player or variant", 503: "No model loaded, or busy"}),
    dependencies=[Depends(inference_slot)],
)
def similar_players(
    service: ServiceDep,
    player_id: Annotated[int, Path(ge=1)],
    k: Annotated[int, Query(ge=1, le=25)] = 8,
    variant: Annotated[str | None, Query()] = None,
) -> SimilarPlayersResponse:
    """Distance is on the preprocessed matrix the model actually sees, and the
    pool is restricted to the same season — a 2014 striker is not a comparison
    for a 2024 one."""
    results = service.similar_players(player_id, k=k, variant=variant)
    return SimilarPlayersResponse(
        player_id=player_id,
        season=results[0]["season"] if results else None,
        results=results,  # type: ignore[arg-type]
    )


@router.get(
    "/players/{player_id}/history",
    response_model=PredictionHistoryResponse,
    summary="Predicted against actual, season by season",
    responses=errors({404: "No such player or variant", 503: "No model loaded, or busy"}),
    dependencies=[Depends(inference_slot)],
)
def prediction_history(
    service: ServiceDep,
    player_id: Annotated[int, Path(ge=1)],
    variant: Annotated[str | None, Query()] = None,
) -> PredictionHistoryResponse:
    """One prediction says what the model thinks; a series says whether it
    tracks a career. Each point states whether the model trained on that
    season, because agreement inside the training range is not evidence."""
    points = service.prediction_history(player_id, variant=variant)
    return PredictionHistoryResponse(
        player_id=player_id,
        variant=service.artifact(variant).variant,
        points=points,  # type: ignore[arg-type]
    )


@router.get(
    "/features/distribution",
    response_model=FeatureDistributionResponse,
    summary="Population quantiles per feature",
    responses=errors({404: "No such variant", 503: "No model loaded"}),
)
def feature_distribution(
    service: ServiceDep, variant: Annotated[str | None, Query()] = None
) -> FeatureDistributionResponse:
    """Lets a client say "92nd percentile for goals per 90" rather than
    normalising against whatever two players happen to be on screen."""
    payload = service.feature_distribution(variant)
    if not payload:
        return FeatureDistributionResponse(
            variant=service.artifact(variant).variant, grid=[], features={}
        )
    return FeatureDistributionResponse(**payload)


@router.get(
    "/players/{player_id}",
    response_model=PlayerResponse,
    summary="A player's record and every season on file",
    responses=errors({404: "No such player or variant", 503: "No model loaded"}),
)
def get_player(
    service: ServiceDep,
    player_id: Annotated[int, Path(ge=1, description="Transfermarkt player id")],
    variant: Annotated[
        str | None,
        Query(description="Whose model inputs `features` holds; default model if omitted"),
    ] = None,
) -> PlayerResponse:
    """`features` are the chosen variant's inputs, so they can seed a what-if
    request for that variant unchanged — prior-value inputs included."""
    return PlayerResponse(**service.player(player_id, variant=variant))


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="Loaded model variants",
    responses=errors({}, validated=False),
)
def list_models(service: ServiceDep) -> ModelListResponse:
    variants = service.variants
    return ModelListResponse(variants=variants, default=variants[0] if variants else None)


@router.get(
    "/models/{variant}",
    response_model=ModelInfoResponse,
    summary="One model's provenance",
    responses=errors({404: "No such variant", 503: "No model loaded"}),
)
def get_model(service: ServiceDep, variant: VariantPath) -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_info(variant))


@router.get(
    "/models/{variant}/metrics",
    response_model=ModelMetricsResponse,
    summary="Held-out metrics, in EUR",
    responses=errors({404: "No such variant", 503: "No model loaded"}),
)
def get_metrics(service: ServiceDep, variant: VariantPath) -> ModelMetricsResponse:
    """Test metrics come from seasons the model never saw. That is the headline."""
    return ModelMetricsResponse(**service.metrics(variant))


@router.get(
    "/models/{variant}/feature-importance",
    response_model=FeatureImportanceResponse,
    summary="Ranked importances, and global SHAP where available",
    responses=errors({404: "No such variant", 503: "No model loaded, or busy"}),
    dependencies=[Depends(inference_slot)],
)
def get_feature_importance(
    service: ServiceDep,
    variant: VariantPath,
    top_n: Annotated[int, Query(ge=1, le=100)] = 20,
    include_shap: Annotated[bool, Query(description="Compute global SHAP (slower)")] = False,
) -> FeatureImportanceResponse:
    """SHAP is opt-in: it samples and re-explains, which takes about a second."""
    payload = service.feature_importance(variant, top_n=top_n)
    if include_shap:
        payload["shap"] = service.global_explanation(variant, top_n=top_n)
    return FeatureImportanceResponse(**payload)
