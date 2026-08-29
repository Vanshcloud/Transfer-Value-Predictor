"""The prediction service, driven without a web server.

That these tests need no TestClient is the point of the layer existing: the
same code path serves HTTP, a batch job and a CLI, so it is tested once, here,
without anything HTTP-shaped in sight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate
from src.feature_engineering.build import TARGET_COLUMN
from src.models.artifact import ModelArtifact, extract_feature_importance, save
from src.models.calibration import calibrate
from src.models.registry import MODEL_REGISTRY, build_pipeline
from src.services.prediction import (
    InvalidFeaturesError,
    ModelNotFoundError,
    ModelUnavailableError,
    PlayerNotFoundError,
    PredictionService,
    SeasonNotFoundError,
    ServiceError,
)

NUMERIC = ["age", "goals", "minutes_played"]
CATEGORICAL = ["position"]
FEATURES = ("age", "goals", "minutes_played", "position")


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for player_id in range(1, 41):
        for season in (2022, 2023, 2024):
            goals = int(rng.integers(0, 25))
            minutes = int(rng.integers(200, 3000))
            age = float(rng.uniform(18, 34))
            rows.append(
                {
                    "player_id": player_id,
                    "season": season,
                    "age": age,
                    "goals": goals,
                    "assists": int(rng.integers(0, 12)),
                    "appearances": int(rng.integers(1, 38)),
                    "minutes_played": minutes,
                    "position": rng.choice(["Attack", "Defender", "Midfield"]),
                    "sub_position": "Centre-Forward",
                    "foot": "right",
                    "height_in_cm": 180.0,
                    "country_of_citizenship": "Brazil",
                    TARGET_COLUMN: float(
                        np.expm1(13 + 0.09 * goals + 0.0004 * minutes - 0.03 * (age - 25) ** 2)
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def artifact(players: pd.DataFrame) -> ModelArtifact:
    pipeline = build_pipeline(MODEL_REGISTRY["lightgbm"], NUMERIC, CATEGORICAL)
    pipeline.fit(players[list(FEATURES)], players[TARGET_COLUMN])
    predictions = pipeline.predict(players[list(FEATURES)])
    metrics = evaluate(players[TARGET_COLUMN], predictions)

    return ModelArtifact(
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


@pytest.fixture
def service(artifact: ModelArtifact, players: pd.DataFrame) -> PredictionService:
    return PredictionService({"performance_only": artifact}, players)


class TestPredictForPlayer:
    def test_it_predicts_in_euros(self, service: PredictionService) -> None:
        result = service.predict_for_player(1)
        assert result.prediction_eur > 10_000  # not a log-space number
        assert result.player_id == 1

    def test_it_reports_the_season_it_used(self, service: PredictionService) -> None:
        """A prediction whose input period is ambiguous is not auditable."""
        assert service.predict_for_player(1).season == 2024
        assert service.predict_for_player(1, season=2022).season == 2022

    def test_omitting_the_season_uses_the_most_recent(self, service: PredictionService) -> None:
        assert service.predict_for_player(1).season == 2024

    def test_an_unknown_player_is_reported_not_guessed(self, service: PredictionService) -> None:
        with pytest.raises(PlayerNotFoundError) as caught:
            service.predict_for_player(999_999)
        assert caught.value.code == "player_not_found"

    def test_a_known_player_in_an_unknown_season_is_distinguished(
        self, service: PredictionService
    ) -> None:
        # A different error from "no such player": the client can tell whether
        # to retry with another season or give up on the id.
        with pytest.raises(SeasonNotFoundError) as caught:
            service.predict_for_player(1, season=1998)
        assert caught.value.code == "season_not_found"

    def test_the_explanation_comes_back_with_the_prediction(
        self, service: PredictionService
    ) -> None:
        result = service.predict_for_player(1)
        assert result.explanation is not None
        assert result.top_positive(3)
        assert all(c["shap_value"] > 0 for c in result.top_positive(3))
        assert all(c["shap_value"] < 0 for c in result.top_negative(3))

    def test_top_n_truncates(self, service: PredictionService) -> None:
        result = service.predict_for_player(1)
        assert len(result.top_positive(2)) <= 2


class TestPredictFromFeatures:
    def test_it_predicts_from_an_explicit_mapping(self, service: PredictionService) -> None:
        result = service.predict_from_features(
            {"age": 24.0, "goals": 15, "minutes_played": 2500, "position": "Attack"}
        )
        assert result.prediction_eur > 10_000
        assert result.player_id is None

    def test_an_unknown_feature_is_rejected_not_dropped(self, service: PredictionService) -> None:
        """Silently dropping a misspelt key answers a question nobody asked."""
        with pytest.raises(InvalidFeaturesError) as caught:
            service.predict_from_features({"age": 24.0, "minutes_playd": 2500})
        assert "minutes_playd" in caught.value.detail  # type: ignore[operator]

    def test_omitted_features_are_imputed_by_the_fitted_pipeline(
        self, service: PredictionService
    ) -> None:
        # Imputed exactly as during training — not zero-filled by the caller.
        result = service.predict_from_features({"goals": 10})
        assert result.prediction_eur > 0

    def test_more_goals_raises_the_prediction(self, service: PredictionService) -> None:
        base = {"age": 25.0, "minutes_played": 2500, "position": "Attack"}
        low = service.predict_from_features({**base, "goals": 1}).prediction_eur
        high = service.predict_from_features({**base, "goals": 22}).prediction_eur
        assert high > low


class TestConfidence:
    def test_it_is_an_interval_not_a_made_up_probability(self, service: PredictionService) -> None:
        confidence = service.predict_for_player(1).confidence
        assert set(confidence) == {
            "level",
            "lower_eur",
            "upper_eur",
            "basis",
            "reference_rows",
        }
        assert "confidence_score" not in confidence

    def test_the_interval_brackets_the_prediction(self, service: PredictionService) -> None:
        result = service.predict_for_player(1)
        assert result.confidence["lower_eur"] < result.prediction_eur
        assert result.confidence["upper_eur"] > result.prediction_eur

    def test_it_says_what_it_was_measured_from(self, service: PredictionService) -> None:
        confidence = service.predict_for_player(1).confidence
        assert "residual quantiles" in confidence["basis"]
        assert confidence["reference_rows"] > 0

    def test_a_model_without_calibration_reports_none_rather_than_inventing_one(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        import dataclasses

        bare = dataclasses.replace(artifact, calibration={})
        service = PredictionService({"performance_only": bare}, players)
        assert service.predict_for_player(1).confidence == {}


class TestPlayerLookup:
    def test_it_returns_every_season_in_order(self, service: PredictionService) -> None:
        player = service.player(1)
        assert [row["season"] for row in player["seasons"]] == [2022, 2023, 2024]

    def test_an_unknown_player_raises(self, service: PredictionService) -> None:
        with pytest.raises(PlayerNotFoundError):
            service.player(999_999)

    def test_it_reports_attributes_from_the_latest_season(self, service: PredictionService) -> None:
        assert service.player(1)["country_of_citizenship"] == "Brazil"


class TestModelSelection:
    def test_an_unknown_variant_is_reported(self, service: PredictionService) -> None:
        with pytest.raises(ModelNotFoundError):
            service.artifact("does_not_exist")

    def test_an_empty_service_is_not_ready(self) -> None:
        service = PredictionService({})
        assert not service.ready
        with pytest.raises(ModelUnavailableError):
            service.artifact()

    def test_omitting_the_variant_picks_a_stable_default(self, service: PredictionService) -> None:
        assert service.artifact().variant == service.artifact(None).variant

    def test_metrics_are_reported_in_euros(self, service: PredictionService) -> None:
        metrics = service.metrics("performance_only")
        assert metrics["test"]["mae_eur"] > 1_000
        assert "mae_log" not in metrics["test"]

    def test_model_info_carries_provenance(self, service: PredictionService) -> None:
        info = service.model_info("performance_only")
        assert info["seed"] == 42
        assert info["feature_columns"] == list(FEATURES)
        assert info["explainable"] is True

    def test_feature_importance_is_ranked(self, service: PredictionService) -> None:
        features = service.feature_importance("performance_only")["features"]
        values = [abs(f["importance"]) for f in features]
        assert values == sorted(values, reverse=True)


class TestLoading:
    def test_it_loads_artifacts_from_a_directory(
        self, artifact: ModelArtifact, tmp_path: object
    ) -> None:
        from pathlib import Path

        directory = Path(str(tmp_path))
        save(artifact, directory)
        service = PredictionService.from_directory(directory)
        assert service.variants == ["performance_only"]

    def test_an_empty_directory_yields_a_service_that_says_so(self, tmp_path: object) -> None:
        from pathlib import Path

        service = PredictionService.from_directory(Path(str(tmp_path)))
        assert not service.ready
        assert service.variants == []

    def test_a_stale_artifact_does_not_stop_the_others_loading(
        self, artifact: ModelArtifact, tmp_path: object
    ) -> None:
        import dataclasses
        from pathlib import Path

        directory = Path(str(tmp_path))
        save(artifact, directory)
        stale = dataclasses.replace(artifact, variant="with_prior_value")
        stale.artifact_version = 99
        save(stale, directory)

        service = PredictionService.from_directory(directory)
        assert service.variants == ["performance_only"]


def test_every_service_error_carries_a_stable_code() -> None:
    """Clients branch on `code`; the message is free to be reworded."""
    for exception in (
        PlayerNotFoundError,
        SeasonNotFoundError,
        ModelNotFoundError,
        ModelUnavailableError,
        InvalidFeaturesError,
    ):
        assert issubclass(exception, ServiceError)
        assert exception.code != ServiceError.code


def test_the_service_imports_no_web_framework() -> None:
    """The layering rule, asserted rather than trusted.

    If this fails, the service has stopped being reusable from a batch job or
    a CLI and has quietly become a web handler.

    Checked against the parsed imports rather than the raw text: the module
    docstring names FastAPI while explaining why it does not import it, and a
    grep cannot tell those two things apart.
    """
    import ast
    import inspect

    import src.services.prediction as module

    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {"fastapi", "starlette", "pydantic"}, sorted(imported)
