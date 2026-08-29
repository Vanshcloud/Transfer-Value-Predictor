"""HTTP transport. Schemas in, service calls out, status codes on the way back.

Every handler here is a few lines: validate, call the service, shape the
response. Anything longer than that belongs in
:mod:`src.services.prediction`, which knows nothing about HTTP and can
therefore be reused by a batch job or a CLI.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query

from api.dependencies import ServiceDep
from api.schemas import (
    ConfidenceSchema,
    ExplanationSchema,
    FeatureImportanceResponse,
    ModelInfoResponse,
    ModelListResponse,
    ModelMetricsResponse,
    ModelSummary,
    PlayerResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(prefix="/api/v1")

VariantPath = Annotated[str, Path(description="performance_only or with_prior_value")]


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict a market value, with an explanation",
    responses={
        404: {"description": "Player, season or model not found"},
        422: {"description": "Malformed body, or an unknown feature name"},
        503: {"description": "No model artifact is loaded"},
    },
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
    "/players/{player_id}",
    response_model=PlayerResponse,
    summary="A player's record and every season on file",
    responses={404: {"description": "No such player"}},
)
def get_player(
    service: ServiceDep,
    player_id: Annotated[int, Path(ge=1, description="Transfermarkt player id")],
) -> PlayerResponse:
    return PlayerResponse(**service.player(player_id))


@router.get("/models", response_model=ModelListResponse, summary="Loaded model variants")
def list_models(service: ServiceDep) -> ModelListResponse:
    variants = service.variants
    return ModelListResponse(variants=variants, default=variants[0] if variants else None)


@router.get(
    "/models/{variant}",
    response_model=ModelInfoResponse,
    summary="One model's provenance",
    responses={404: {"description": "No such variant"}, 503: {"description": "No model loaded"}},
)
def get_model(service: ServiceDep, variant: VariantPath) -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_info(variant))


@router.get(
    "/models/{variant}/metrics",
    response_model=ModelMetricsResponse,
    summary="Held-out metrics, in EUR",
    responses={404: {"description": "No such variant"}, 503: {"description": "No model loaded"}},
)
def get_metrics(service: ServiceDep, variant: VariantPath) -> ModelMetricsResponse:
    """Test metrics come from seasons the model never saw. That is the headline."""
    return ModelMetricsResponse(**service.metrics(variant))


@router.get(
    "/models/{variant}/feature-importance",
    response_model=FeatureImportanceResponse,
    summary="Ranked importances, and global SHAP where available",
    responses={404: {"description": "No such variant"}, 503: {"description": "No model loaded"}},
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
