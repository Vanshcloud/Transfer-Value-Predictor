"""The API against the real artifacts, through the real lifespan.

The unit tests substitute the service and skip lifespan, which is what keeps
them fast. This file covers the part that then goes untested: that the app
actually starts, finds artifacts on disk, and serves them.

Marked ``integration``: skips when no model has been trained.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    if not list((PROJECT_ROOT / "models").glob("*.joblib")):
        pytest.skip("run scripts/train_models.py first")

    # As a context manager, so the real lifespan handler runs.
    with TestClient(create_app()) as live:
        yield live


def test_the_app_starts_and_loads_models(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["ready"] is True
    assert set(body["models_loaded"]) == {"performance_only", "with_prior_value"}


def test_both_variants_are_servable(client: TestClient) -> None:
    for variant in ("performance_only", "with_prior_value"):
        response = client.get(f"/api/v1/models/{variant}")
        assert response.status_code == 200
        assert response.json()["variant"] == variant


def test_a_real_prediction_round_trips(client: TestClient) -> None:
    player_id = 28003

    listed = client.post("/api/v1/predict", json={"player_id": player_id})
    if listed.status_code == 404:
        pytest.skip(f"player {player_id} not in this refresh of the dataset")

    body = listed.json()
    assert body["prediction_eur"] > 10_000
    assert body["explanation"]["top_positive_features"]
    assert body["confidence"]["reference_rows"] > 0


def test_the_prior_value_variant_predicts_too(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={"player_id": 28003, "variant": "with_prior_value"},
    )
    if response.status_code == 404:
        pytest.skip("player not in this refresh")
    assert response.json()["variant"] == "with_prior_value"


def test_metrics_match_the_artifact_on_disk(client: TestClient) -> None:
    """The API must report the model's real numbers, not recomputed ones."""
    from src.models.artifact import load

    artifact = load(next((PROJECT_ROOT / "models").glob("performance_only__*.joblib")))
    served = client.get("/api/v1/models/performance_only/metrics").json()

    assert served["test"]["mae_eur"] == pytest.approx(artifact.test.mae)
    assert served["test"]["r2"] == pytest.approx(artifact.test.r2)


def test_the_confidence_interval_brackets_real_predictions(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "features": {
                "age": 24.0,
                "goals": 14,
                "minutes_played": 2500,
                "appearances": 30,
                "position": "Attack",
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    confidence = body["confidence"]

    assert confidence["lower_eur"] < body["prediction_eur"] < confidence["upper_eur"]
    assert confidence["level"] == 0.8


def test_explanations_are_additive_in_log_space_over_http(client: TestClient) -> None:
    """The property the contract promises clients, checked end to end."""
    response = client.post(
        "/api/v1/predict",
        json={"features": {"age": 24.0, "goals": 14, "minutes_played": 2500}, "top_n": 25},
    )
    body = response.json()
    contributions = (
        body["explanation"]["top_positive_features"] + body["explanation"]["top_negative_features"]
    )
    for contribution in contributions:
        assert contribution["effect_multiplier"] == pytest.approx(
            np.exp(contribution["shap_value"])
        )


def test_global_shap_is_available_over_http(client: TestClient) -> None:
    body = client.get(
        "/api/v1/models/performance_only/feature-importance?include_shap=true&top_n=5"
    ).json()
    assert len(body["features"]) == 5
    assert body["shap"]["features"]


def test_the_openapi_document_describes_the_live_app(client: TestClient) -> None:
    spec = client.get("/api/v1/openapi.json").json()
    assert spec["info"]["title"] == "Transfer Value Predictor"
    assert "/api/v1/predict" in spec["paths"]
