"""The HTTP layer: status codes, envelopes and the OpenAPI document.

The service is substituted rather than loaded from disk, so these tests are
about transport — did the right status code come back, in the documented
shape — and not about whether the model is any good. That separation is only
possible because the service takes no web framework.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_service
from api.main import create_app
from src.evaluation.metrics import evaluate
from src.feature_engineering.build import TARGET_COLUMN
from src.models.artifact import ModelArtifact, extract_feature_importance
from src.models.calibration import calibrate
from src.models.registry import MODEL_REGISTRY, build_pipeline
from src.services.prediction import PredictionService

NUMERIC = ["age", "goals", "minutes_played"]
CATEGORICAL = ["position"]
FEATURES = ("age", "goals", "minutes_played", "position")


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    # Enough rows that LightGBM actually splits. With a handful, it fits a
    # near-constant model whose explanations have no positive contributions at
    # all, and the tests then assert against a degenerate model rather than the
    # API.
    rng = np.random.default_rng(9)
    rows = []
    for player_id in range(101, 141):
        for season in (2023, 2024):
            goals = int(rng.integers(0, 25))
            minutes = int(rng.integers(200, 3000))
            age = float(rng.uniform(19, 33))
            rows.append(
                {
                    "player_id": player_id,
                    "season": season,
                    "age": age,
                    "goals": goals,
                    "assists": int(rng.integers(0, 10)),
                    "appearances": int(rng.integers(1, 38)),
                    "minutes_played": minutes,
                    "position": rng.choice(["Attack", "Defender", "Midfield"]),
                    "sub_position": "Centre-Forward",
                    "foot": "right",
                    "height_in_cm": 182.0,
                    "country_of_citizenship": "Brazil",
                    TARGET_COLUMN: float(
                        np.expm1(
                            13
                            + 0.09 * goals
                            + 0.0004 * minutes
                            - 0.03 * (age - 25) ** 2
                            + rng.normal(0, 0.2)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def service(players: pd.DataFrame) -> PredictionService:
    pipeline = build_pipeline(MODEL_REGISTRY["lightgbm"], NUMERIC, CATEGORICAL)
    pipeline.fit(players[list(FEATURES)], players[TARGET_COLUMN])
    predictions = pipeline.predict(players[list(FEATURES)])
    metrics = evaluate(players[TARGET_COLUMN], predictions)

    artifact = ModelArtifact(
        variant="performance_only",
        model_name="lightgbm",
        params={},
        pipeline=pipeline,
        validation=metrics,
        test=metrics,
        feature_columns=FEATURES,
        target_column=TARGET_COLUMN,
        feature_importance=extract_feature_importance(pipeline),
        dataset={"rows": len(players)},
        split={"strategy": "temporal"},
        seed=42,
        calibration=calibrate(
            players[TARGET_COLUMN].to_numpy(dtype=float), np.asarray(predictions)
        ),
    )
    names = {pid: f"Player {pid}" for pid in players["player_id"].unique()}
    return PredictionService({"performance_only": artifact}, players, names)


def make_client(service: PredictionService) -> Iterator[TestClient]:
    """A client with the service substituted and the real lifespan skipped.

    Two deliberate choices, both learned the hard way:

    * The service is injected through ``dependency_overrides``, not by setting
      ``app.state``. Entering TestClient's context manager runs the lifespan
      handler, which overwrites app.state with models loaded from disk — so a
      substituted service set that way is silently discarded.
    * The client is not used as a context manager, so lifespan never runs.
      These are unit tests of the transport layer: they should not need a
      trained artifact on disk or a 37,000-row table in memory.

    The real lifespan is covered by tests/integration/test_api_live.py.
    """
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client(service: PredictionService) -> Iterator[TestClient]:
    yield from make_client(service)


@pytest.fixture
def empty_client() -> Iterator[TestClient]:
    yield from make_client(PredictionService({}))


class TestHealth:
    def test_it_is_unversioned(self, client: TestClient) -> None:
        # Load balancers should not need to know about API versions.
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/health").status_code == 404

    def test_it_reports_ready_with_a_model(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["ready"] is True
        assert body["models_loaded"] == ["performance_only"]

    def test_it_reports_degraded_but_still_200_without_a_model(
        self, empty_client: TestClient
    ) -> None:
        # A health check that reports down for a missing model would have an
        # orchestrator restart the process forever.
        response = empty_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["ready"] is False


class TestPredict:
    def test_predicting_by_player_id(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={"player_id": 101})
        assert response.status_code == 200

        body = response.json()
        assert body["prediction_eur"] > 10_000
        assert body["player_id"] == 101
        assert body["season"] == 2024
        assert body["model"]["name"] == "lightgbm"

    def test_predicting_from_features(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict",
            json={
                "features": {"age": 24.0, "goals": 18, "minutes_played": 2600, "position": "Attack"}
            },
        )
        assert response.status_code == 200
        assert response.json()["player_id"] is None

    def test_the_response_carries_both_contribution_directions(self, client: TestClient) -> None:
        body = client.post("/api/v1/predict", json={"player_id": 101}).json()
        explanation = body["explanation"]

        assert explanation["top_positive_features"]
        assert all(c["shap_value"] > 0 for c in explanation["top_positive_features"])
        assert all(c["shap_value"] < 0 for c in explanation["top_negative_features"])

    def test_every_contribution_carries_the_multiplicative_reading(
        self, client: TestClient
    ) -> None:
        # shap_value is log-space and must never be summed into euros;
        # effect_multiplier is the exact reading a client can use.
        body = client.post("/api/v1/predict", json={"player_id": 101}).json()
        for contribution in body["explanation"]["top_positive_features"]:
            assert contribution["effect_multiplier"] == pytest.approx(
                np.exp(contribution["shap_value"])
            )

    def test_top_n_is_honoured(self, client: TestClient) -> None:
        body = client.post("/api/v1/predict", json={"player_id": 101, "top_n": 2}).json()
        assert len(body["explanation"]["top_positive_features"]) <= 2

    def test_confidence_is_an_interval_with_a_stated_basis(self, client: TestClient) -> None:
        body = client.post("/api/v1/predict", json={"player_id": 101}).json()
        confidence = body["confidence"]

        assert confidence["lower_eur"] < body["prediction_eur"] < confidence["upper_eur"]
        assert 0 < confidence["level"] <= 1
        assert confidence["basis"]
        assert confidence["reference_rows"] > 0

    def test_a_specific_season_can_be_requested(self, client: TestClient) -> None:
        body = client.post("/api/v1/predict", json={"player_id": 101, "season": 2023}).json()
        assert body["season"] == 2023

    def test_an_unknown_player_is_404_in_the_documented_envelope(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={"player_id": 999999})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "player_not_found"

    def test_an_unknown_season_is_distinguished_from_an_unknown_player(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/v1/predict", json={"player_id": 101, "season": 1990})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "season_not_found"

    def test_an_unknown_feature_is_422_naming_the_field(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={"features": {"minutes_playd": 100}})
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "validation_error"
        assert "minutes_playd" in body["error"]["detail"]

    def test_supplying_both_inputs_is_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={"player_id": 101, "features": {"age": 25}})
        assert response.status_code == 422

    def test_supplying_neither_input_is_422(self, client: TestClient) -> None:
        assert client.post("/api/v1/predict", json={}).status_code == 422

    def test_an_unknown_body_field_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={"player_id": 101, "typo_field": 1})
        assert response.status_code == 422

    def test_a_malformed_body_returns_useful_field_level_detail(self, client: TestClient) -> None:
        """A bare 'Unprocessable Entity' forces the client to guess."""
        response = client.post("/api/v1/predict", json={"player_id": "not-a-number"})
        assert response.status_code == 422

        detail = response.json()["error"]["detail"]
        assert isinstance(detail, list)
        assert any("player_id" in str(item["loc"]) for item in detail)

    def test_predicting_without_a_model_is_503(self, empty_client: TestClient) -> None:
        response = empty_client.post("/api/v1/predict", json={"player_id": 101})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"


class TestPlayers:
    def test_it_returns_the_seasons_on_file(self, client: TestClient) -> None:
        body = client.get("/api/v1/players/101").json()
        assert body["player_id"] == 101
        assert [row["season"] for row in body["seasons"]] == [2023, 2024]

    def test_an_unknown_player_is_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/players/999999")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "player_not_found"

    def test_a_non_numeric_id_is_422(self, client: TestClient) -> None:
        assert client.get("/api/v1/players/abc").status_code == 422


class TestModels:
    def test_listing_models(self, client: TestClient) -> None:
        body = client.get("/api/v1/models").json()
        assert body["variants"] == ["performance_only"]
        assert body["default"] == "performance_only"

    def test_model_info_carries_provenance(self, client: TestClient) -> None:
        body = client.get("/api/v1/models/performance_only").json()
        assert body["seed"] == 42
        assert body["split"]["strategy"] == "temporal"
        assert body["feature_columns"] == list(FEATURES)

    def test_metrics_are_in_euros(self, client: TestClient) -> None:
        body = client.get("/api/v1/models/performance_only/metrics").json()
        assert body["test"]["mae_eur"] > 1_000

    def test_feature_importance_is_ranked(self, client: TestClient) -> None:
        body = client.get("/api/v1/models/performance_only/feature-importance").json()
        values = [abs(f["importance"]) for f in body["features"]]
        assert values == sorted(values, reverse=True)

    def test_shap_is_opt_in(self, client: TestClient) -> None:
        # It samples and re-explains, which costs about a second.
        without = client.get("/api/v1/models/performance_only/feature-importance").json()
        assert without["shap"] == {}

        with_shap = client.get(
            "/api/v1/models/performance_only/feature-importance?include_shap=true"
        ).json()
        assert with_shap["shap"]["features"]

    def test_an_unknown_variant_is_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/models/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"


class TestContract:
    def test_every_documented_endpoint_exists(self, client: TestClient) -> None:
        paths = set(client.get("/api/v1/openapi.json").json()["paths"])
        assert paths == {
            "/health",
            "/api/v1/predict",
            "/api/v1/players",
            "/api/v1/players/{player_id}",
            "/api/v1/players/{player_id}/similar",
            "/api/v1/players/{player_id}/history",
            "/api/v1/features/distribution",
            "/api/v1/models",
            "/api/v1/models/{variant}",
            "/api/v1/models/{variant}/metrics",
            "/api/v1/models/{variant}/feature-importance",
        }

    def test_the_openapi_document_is_served_under_the_version(self, client: TestClient) -> None:
        assert client.get("/api/v1/openapi.json").status_code == 200

    def test_interactive_docs_are_served(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200

    def test_schemas_carry_descriptions_not_just_types(self, client: TestClient) -> None:
        """An OpenAPI document with bare types is a type checker, not docs."""
        schemas = client.get("/api/v1/openapi.json").json()["components"]["schemas"]
        confidence = schemas["ConfidenceSchema"]["properties"]
        assert confidence["basis"]["description"]
        assert confidence["reference_rows"]["description"]

    def test_the_predict_schema_documents_both_request_shapes(self, client: TestClient) -> None:
        schema = client.get("/api/v1/openapi.json").json()["components"]["schemas"][
            "PredictRequest"
        ]
        assert len(schema["examples"]) == 2

    def test_every_error_uses_one_envelope(self, client: TestClient) -> None:
        responses = [
            client.post("/api/v1/predict", json={"player_id": 999999}),
            client.post("/api/v1/predict", json={}),
            client.get("/api/v1/models/nope"),
            client.get("/api/v1/players/999999"),
        ]
        for response in responses:
            body = response.json()
            assert set(body) == {"error"}
            assert set(body["error"]) == {"code", "message", "detail"}


class TestSearchEndpoint:
    def test_it_finds_a_player_by_name(self, client: TestClient) -> None:
        body = client.get("/api/v1/players?q=player%20101").json()
        assert body["query"] == "player 101"
        assert body["results"][0]["player_id"] == 101

    def test_it_is_case_insensitive(self, client: TestClient) -> None:
        assert client.get("/api/v1/players?q=PLAYER%20101").json()["results"]

    def test_an_empty_query_is_rejected_by_the_schema(self, client: TestClient) -> None:
        assert client.get("/api/v1/players?q=").status_code == 422

    def test_the_limit_is_bounded(self, client: TestClient) -> None:
        assert client.get("/api/v1/players?q=player&limit=999").status_code == 422

    def test_results_say_whether_they_can_be_predicted(self, client: TestClient) -> None:
        results = client.get("/api/v1/players?q=player&limit=5").json()["results"]
        assert all("predictable" in row for row in results)


class TestSimilarEndpoint:
    def test_it_returns_neighbours(self, client: TestClient) -> None:
        body = client.get("/api/v1/players/101/similar?k=3").json()
        assert body["player_id"] == 101
        assert len(body["results"]) == 3
        assert 101 not in [row["player_id"] for row in body["results"]]

    def test_neighbours_are_ordered_by_distance(self, client: TestClient) -> None:
        results = client.get("/api/v1/players/101/similar?k=5").json()["results"]
        distances = [row["distance"] for row in results]
        assert distances == sorted(distances)

    def test_an_unknown_player_is_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/players/999999/similar")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "player_not_found"

    def test_k_is_bounded(self, client: TestClient) -> None:
        assert client.get("/api/v1/players/101/similar?k=500").status_code == 422


class TestWhatIfSupport:
    def test_the_player_payload_seeds_a_what_if_form(self, client: TestClient) -> None:
        """The dashboard needs the real starting point, not a guess at it."""
        body = client.get("/api/v1/players/101").json()
        assert body["features"]
        assert body["name"]

    def test_the_seeded_features_reproduce_the_stored_prediction(self, client: TestClient) -> None:
        # If the seed did not round-trip, every what-if would start from a
        # different player than the one on screen.
        player = client.get("/api/v1/players/101").json()
        stored = client.post("/api/v1/predict", json={"player_id": 101}).json()
        replayed = client.post("/api/v1/predict", json={"features": player["features"]}).json()

        assert replayed["prediction_eur"] == pytest.approx(stored["prediction_eur"])

    def test_changing_one_feature_moves_the_prediction(self, client: TestClient) -> None:
        """What-if is only useful if the sliders actually do something.

        Compared across the training range rather than by a fixed increment:
        adding 15 to an already-high value lands past every split the trees
        learned, so the prediction legitimately does not move — which would
        fail this test on a model that is working correctly.
        """
        player = client.get("/api/v1/players/101").json()
        features = dict(player["features"])

        low = client.post("/api/v1/predict", json={"features": {**features, "goals": 0}}).json()
        high = client.post("/api/v1/predict", json={"features": {**features, "goals": 22}}).json()

        assert high["prediction_eur"] > low["prediction_eur"]


class TestHistoryEndpoint:
    def test_it_returns_a_point_per_season(self, client: TestClient) -> None:
        body = client.get("/api/v1/players/101/history").json()
        assert body["player_id"] == 101
        assert [point["season"] for point in body["points"]] == [2023, 2024]

    def test_each_point_flags_whether_the_model_saw_it(self, client: TestClient) -> None:
        for point in client.get("/api/v1/players/101/history").json()["points"]:
            assert "in_training_range" in point
            assert "held_out" in point

    def test_it_agrees_with_the_predict_endpoint(self, client: TestClient) -> None:
        points = client.get("/api/v1/players/101/history").json()["points"]
        latest = max(points, key=lambda p: p["season"])
        single = client.post("/api/v1/predict", json={"player_id": 101}).json()

        assert latest["predicted_eur"] == pytest.approx(single["prediction_eur"])

    def test_an_unknown_player_is_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/players/999999/history").status_code == 404


class TestDistributionEndpoint:
    def test_it_returns_a_quantile_grid(self, client: TestClient) -> None:
        body = client.get("/api/v1/features/distribution").json()
        assert body["grid"][0] == 0.0
        assert body["grid"][-1] == 1.0
        assert body["features"]

    def test_quantiles_are_monotonic(self, client: TestClient) -> None:
        body = client.get("/api/v1/features/distribution").json()
        for feature in body["features"].values():
            assert feature["quantiles"] == sorted(feature["quantiles"])

    def test_without_a_model_it_is_503(self, empty_client: TestClient) -> None:
        assert empty_client.get("/api/v1/features/distribution").status_code == 503
