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


class TestNothingImportantIsSilentlyEmpty:
    """The gap that let two high-severity bugs survive four audits.

    Both `similar_players` and `prediction_history` returned `[]` — a `200`, a
    valid shape, a dashboard panel reading "No comparable seasons found". Every
    test asserted the *shape* of the response and none asserted that anything
    was in it, so the suite stayed green while a third of the panel got nothing:

      * comparables were dead for all 8,709 players whose most recent row is
        the unlabelled current season (32.8%), because the anchor season had no
        labelled pool;
      * 14,402 of 24,411 players (59%) had an empty career chart, because the
        service required all 54 features to be non-null in a pipeline that
        imputes.

    Shape assertions cannot catch that. These run against the real artifacts
    and the real panel and assert the thing a user would actually notice.
    """

    def _some_player_with_a_current_season(self, client: TestClient) -> int:
        """A player whose latest row is the season being played — the exact
        population both bugs silently excluded."""
        service = client.app.state.service
        current = service._players[~service._players["has_label"].fillna(True)]
        if current.empty:
            pytest.skip("no current-season rows in this build")
        return int(current.iloc[0]["player_id"])

    def test_an_active_player_gets_comparables(self, client: TestClient) -> None:
        player_id = self._some_player_with_a_current_season(client)
        results = client.get(f"/api/v1/players/{player_id}/similar").json()["results"]
        assert results, f"player {player_id} got no comparables at all"
        assert all(row["market_value_in_eur"] > 0 for row in results)

    def test_an_active_player_gets_a_career_history(self, client: TestClient) -> None:
        player_id = self._some_player_with_a_current_season(client)
        points = client.get(f"/api/v1/players/{player_id}/history").json()["points"]
        assert points, f"player {player_id} got an empty career history"

    def test_history_covers_every_labelled_season_on_record(self, client: TestClient) -> None:
        """Not "some points" — every labelled season. The old bug dropped each
        player's first season, which is the one a career chart most needs."""
        service = client.app.state.service
        panel = service._players
        labelled = panel[panel["has_label"].fillna(True)]
        counts = labelled.groupby("player_id").size()
        player_id = int(counts[counts >= 5].index[0])

        points = client.get(f"/api/v1/players/{player_id}/history").json()["points"]
        expected = sorted(int(s) for s in labelled[labelled["player_id"] == player_id]["season"])
        assert [point["season"] for point in points] == expected

    def test_the_panel_is_overwhelmingly_servable(self, client: TestClient) -> None:
        """A population check, not a spot check. A regression that empties one
        endpoint for a third of players passes every single-player test."""
        service = client.app.state.service
        artifact = service.artifact("performance_only")
        usable = service._rows_for_variant(artifact)
        panel = service._players
        labelled_players = panel[panel["has_label"].fillna(True)]["player_id"].nunique()

        covered = usable["player_id"].nunique()
        assert (
            covered == labelled_players
        ), f"only {covered:,} of {labelled_players:,} labelled players are servable"
