"""The prediction service, driven without a web server.

That these tests need no TestClient is the point of the layer existing: the
same code path serves HTTP, a batch job and a CLI, so it is tested once, here,
without anything HTTP-shaped in sight.
"""

from __future__ import annotations

from pathlib import Path

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


NAMES = {player_id: f"Player {player_id:03d}" for player_id in range(1, 41)}
NAMES[7] = "Erling Test"


@pytest.fixture
def service(artifact: ModelArtifact, players: pd.DataFrame) -> PredictionService:
    return PredictionService({"performance_only": artifact}, players, NAMES)


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


class TestFeatureValueValidation:
    """Values, not names.

    An unknown *name* was always rejected. An unknown-shaped *value* went into
    the fitted pipeline and surfaced as a ValueError from inside a scikit-learn
    transformer — which the API turned into a bare 500 for input a caller
    typed. These run at the service level because the rule belongs to the
    service: a batch job passing a negative goal count is just as wrong as an
    HTTP client doing it.
    """

    @pytest.mark.parametrize(
        ("features", "fragment"),
        [
            ({"age": "twenty-five"}, "expected a number"),
            ({"age": {"$ne": 1}}, "expected a single value"),
            ({"age": [1, 2, 3]}, "expected a single value"),
            ({"age": (1, 2)}, "expected a single value"),
            ({"age": True}, "got a boolean"),
            ({"age": float("nan")}, "finite"),
            ({"age": float("inf")}, "finite"),
            ({"goals": -1}, "cannot be negative"),
            ({"minutes_played": -0.5}, "cannot be negative"),
            ({"position": "x" * 201}, "category longer than"),
            # F8: each of these was a 500 or a confident 200.
            ({"goals": 10**400}, "too large"),
            ({"goals": 2.5}, "whole number"),
            ({"goals": 1e308}, "expected 0-"),
            ({"minutes_played": 10**9}, "expected 0-"),
            ({"position": 5}, "expected text"),
            ({"position": 10**400}, "expected text"),
        ],
    )
    def test_a_bad_value_raises_rather_than_reaching_the_model(
        self, service: PredictionService, features: dict[str, object], fragment: str
    ) -> None:
        with pytest.raises(InvalidFeaturesError) as caught:
            service.predict_from_features(features)
        assert fragment in caught.value.message
        assert caught.value.code == "validation_error"
        assert next(iter(features)) in caught.value.message

    def test_all_bad_values_are_reported_at_once(self, service: PredictionService) -> None:
        """One round trip should be enough to fix the whole request."""
        with pytest.raises(InvalidFeaturesError) as caught:
            service.predict_from_features({"age": "x", "goals": -3})
        assert len(caught.value.detail) == 2  # type: ignore[arg-type]

    def test_zero_is_valid_for_a_non_negative_feature(self, service: PredictionService) -> None:
        """The bound is `< 0`. A player with no goals is ordinary, not invalid."""
        assert service.predict_from_features({"goals": 0}).prediction_eur > 0

    def test_a_whole_number_written_as_a_float_is_a_count(self, service: PredictionService) -> None:
        """JSON clients send 3.0 for three; that is not a fractional goal."""
        assert service.predict_from_features({"goals": 3.0}).prediction_eur > 0

    def test_a_bad_value_is_quoted_back_without_the_whole_payload(
        self, service: PredictionService
    ) -> None:
        with pytest.raises(InvalidFeaturesError) as caught:
            service.predict_from_features({"age": "x" * 10_000})
        assert len(caught.value.message) < 200

    def test_an_explicit_none_still_means_absent(self, service: PredictionService) -> None:
        assert service.predict_from_features({"age": None}).prediction_eur > 0

    def test_a_long_category_under_the_cap_is_accepted(self, service: PredictionService) -> None:
        """Unknown categories are the encoder's job; only absurd ones are ours."""
        assert service.predict_from_features({"position": "x" * 199}).prediction_eur > 0

    def test_numeric_columns_come_from_the_fitted_preprocessor(
        self, service: PredictionService
    ) -> None:
        """Re-listing them here would let the check disagree with the model."""
        numeric = service._numeric_features(service.artifact())
        assert "age" in numeric and "goals" in numeric
        assert "position" not in numeric

    def test_a_valid_call_is_unchanged(self, service: PredictionService) -> None:
        """Validation must not move the number it guards."""
        features = {"age": 24.0, "goals": 15, "minutes_played": 2500, "position": "Attack"}
        assert service.predict_from_features(features).prediction_eur == pytest.approx(
            service.predict_from_features(features).prediction_eur
        )

    def test_the_backstop_turns_a_pipeline_refusal_into_a_bad_request(
        self, service: PredictionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_validate_values catches every malformed value seen so far, but the
        fitted pipeline is the only thing that knows its own full expectations.
        A value it rejects is a bad request, not a broken server — and without
        the backstop the caller gets the bare 500 this whole class exists to
        remove. Forced here, because by construction nothing else reaches it.
        """
        import src.services.prediction as module

        def refuse(*_: object, **__: object) -> object:
            raise ValueError("could not convert string to float: something")

        monkeypatch.setattr(module, "predict", refuse)

        with pytest.raises(InvalidFeaturesError) as caught:
            service.predict_from_features({"age": 25.0})
        assert caught.value.code == "validation_error"
        assert "could not be evaluated" in caught.value.message


class TestConfidence:
    def test_it_is_an_interval_not_a_made_up_probability(self, service: PredictionService) -> None:
        confidence = service.predict_for_player(1).confidence
        assert set(confidence) == {
            "level",
            "lower_eur",
            "upper_eur",
            "basis",
            "reference_rows",
            "measured_coverage",
        }
        assert "confidence_score" not in confidence

    def test_the_nominal_level_is_reported_beside_the_level_it_achieved(
        self, service: PredictionService
    ) -> None:
        """A response carrying only the nominal level quotes its own intention
        as a result. This fixture's calibration was never checked against a
        later season, so the measured figure is null rather than invented."""
        confidence = service.predict_for_player(1).confidence
        assert confidence["level"] == pytest.approx(0.8)
        assert confidence["measured_coverage"] is None

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

    # F19: lexicographic file order used to decide which artifact served.

    def test_a_misleadingly_named_file_does_not_take_over(
        self, artifact: ModelArtifact, tmp_path: Path
    ) -> None:
        import dataclasses
        import shutil

        save(artifact, tmp_path)
        impostor = dataclasses.replace(artifact, model_name="impostor")
        save(impostor, tmp_path / "elsewhere")
        # Sorts after the real file, so name order would have served it.
        shutil.copy(
            tmp_path / "elsewhere" / "performance_only__impostor.joblib",
            tmp_path / "performance_only__zzz.joblib",
        )
        service = PredictionService.from_directory(tmp_path)
        assert service.artifact().model_name == artifact.model_name

    def test_of_two_valid_artifacts_the_manifest_decides(
        self, artifact: ModelArtifact, tmp_path: Path
    ) -> None:
        import dataclasses

        save(dataclasses.replace(artifact, model_name="zzz_older"), tmp_path)
        save(dataclasses.replace(artifact, model_name="aaa_newer"), tmp_path)
        assert PredictionService.from_directory(tmp_path).artifact().model_name == "aaa_newer"

    def test_artifacts_without_a_manifest_are_not_guessed_between(
        self, artifact: ModelArtifact, tmp_path: Path
    ) -> None:
        from src.models.artifact import MANIFEST_NAME

        save(artifact, tmp_path)
        (tmp_path / MANIFEST_NAME).unlink()
        assert not PredictionService.from_directory(tmp_path).ready

    def test_a_file_that_no_longer_matches_its_checksum_is_skipped(
        self, artifact: ModelArtifact, tmp_path: Path
    ) -> None:
        path = save(artifact, tmp_path)
        path.write_bytes(path.read_bytes() + b"\0")
        assert not PredictionService.from_directory(tmp_path).ready

    def test_a_manifest_entry_holding_another_variant_is_skipped(
        self, artifact: ModelArtifact, tmp_path: Path
    ) -> None:
        import json

        from src.models.artifact import MANIFEST_NAME

        save(artifact, tmp_path)
        manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
        manifest["variants"]["with_prior_value"] = manifest["variants"].pop("performance_only")
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
        assert not PredictionService.from_directory(tmp_path).ready

    def test_a_model_the_player_tables_cannot_feed_is_refused(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """F10: an incompatible model/data pair is refused at start-up, not
        discovered one failed lookup at a time."""
        service = PredictionService({"performance_only": artifact}, players.drop(columns="goals"))
        assert not service.ready


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


class TestSearch:
    def test_it_matches_case_insensitively(self, service: PredictionService) -> None:
        assert service.search_players("erling")[0]["player_id"] == 7
        assert service.search_players("ERLING")[0]["player_id"] == 7

    def test_it_matches_a_substring(self, service: PredictionService) -> None:
        assert service.search_players("rling Te")[0]["name"] == "Erling Test"

    def test_an_empty_query_returns_nothing_rather_than_everything(
        self, service: PredictionService
    ) -> None:
        assert service.search_players("   ") == []

    def test_results_are_limited(self, service: PredictionService) -> None:
        assert len(service.search_players("Player", limit=3)) == 3

    def test_a_result_says_whether_it_can_be_predicted(self, service: PredictionService) -> None:
        # A search result that leads to a 404 is worse than no result.
        assert all(row["predictable"] for row in service.search_players("Player"))

    def test_modellable_players_are_ranked_first(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        names = {**NAMES, 9999: "Player unknown"}
        service = PredictionService({"performance_only": artifact}, players, names)
        results = service.search_players("Player", limit=50)
        assert results[-1]["player_id"] == 9999
        assert results[-1]["predictable"] is False

    def test_search_without_names_reports_rather_than_returning_empty(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        service = PredictionService({"performance_only": artifact}, players)
        with pytest.raises(PlayerNotFoundError):
            service.search_players("anything")


class TestSimilarPlayers:
    def test_it_returns_neighbours_excluding_the_query(self, service: PredictionService) -> None:
        results = service.similar_players(1, k=5)
        assert 1 not in [row["player_id"] for row in results]
        assert len(results) == 5

    def test_results_are_ordered_by_distance(self, service: PredictionService) -> None:
        distances = [row["distance"] for row in service.similar_players(1, k=5)]
        assert distances == sorted(distances)

    def test_comparisons_stay_within_one_season(self, service: PredictionService) -> None:
        """Market conditions differ across years; a 2022 season is not a
        comparison for a 2024 one."""
        results = service.similar_players(1, k=5)
        assert {row["season"] for row in results} == {2024}

    def test_it_carries_names_when_they_are_loaded(self, service: PredictionService) -> None:
        assert all(row["name"] for row in service.similar_players(1, k=3))

    def test_an_unknown_player_raises(self, service: PredictionService) -> None:
        with pytest.raises(PlayerNotFoundError):
            service.similar_players(999_999)


class TestFeatureSeed:
    def test_it_returns_every_feature_the_model_expects(self, service: PredictionService) -> None:
        # What-if starts from the real values, not a guess at what is needed.
        features = service.features_for(1)
        assert set(features) == set(FEATURES)

    def test_the_seeded_features_reproduce_the_stored_prediction(
        self, service: PredictionService
    ) -> None:
        """The seed must round-trip, or what-if starts from a different player."""
        stored = service.predict_for_player(1).prediction_eur
        replayed = service.predict_from_features(service.features_for(1)).prediction_eur
        assert replayed == pytest.approx(stored)

    def test_a_specific_season_can_be_seeded(self, service: PredictionService) -> None:
        assert service.features_for(1, season=2022) != service.features_for(1, season=2024)

    def test_an_unknown_season_raises(self, service: PredictionService) -> None:
        with pytest.raises(SeasonNotFoundError):
            service.features_for(1, season=1990)

    def test_the_player_payload_carries_the_seed(self, service: PredictionService) -> None:
        assert set(service.player(1)["features"]) == set(FEATURES)

    def test_the_player_payload_carries_a_name(self, service: PredictionService) -> None:
        assert service.player(7)["name"] == "Erling Test"


class TestPredictionHistory:
    def test_it_covers_every_modellable_season(self, service: PredictionService) -> None:
        history = service.prediction_history(1)
        assert [point["season"] for point in history] == [2022, 2023, 2024]

    def test_each_point_carries_prediction_and_truth(self, service: PredictionService) -> None:
        for point in service.prediction_history(1):
            assert point["predicted_eur"] > 0
            assert point["actual_eur"] > 0
            assert point["error_eur"] == pytest.approx(point["predicted_eur"] - point["actual_eur"])

    def test_predictions_match_the_single_season_endpoint(self, service: PredictionService) -> None:
        """Two paths to the same number must not disagree."""
        history = {p["season"]: p["predicted_eur"] for p in service.prediction_history(1)}
        single = service.predict_for_player(1, season=2023)
        assert history[2023] == pytest.approx(single.prediction_eur)

    def test_points_say_whether_the_model_trained_on_them(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        # Agreement inside the training range is not evidence, so the flag has
        # to travel with the point rather than be inferred by the reader.
        import dataclasses

        split = {"strategy": "temporal", "train_end_season": 2022, "test_start_season": 2024}
        service = PredictionService(
            {"performance_only": dataclasses.replace(artifact, split=split)},
            players,
            NAMES,
        )
        history = {p["season"]: p for p in service.prediction_history(1)}

        assert history[2022]["in_training_range"] is True
        assert history[2022]["held_out"] is False
        assert history[2024]["held_out"] is True

    def test_an_unknown_player_raises(self, service: PredictionService) -> None:
        with pytest.raises(PlayerNotFoundError):
            service.prediction_history(999_999)


class TestFeatureDistribution:
    def test_it_reports_quantiles_per_numeric_feature(self, service: PredictionService) -> None:
        distribution = service.feature_distribution()
        assert distribution["features"]
        assert "goals" in distribution["features"]

    def test_quantiles_are_monotonic(self, service: PredictionService) -> None:
        """A radar axis built on a non-monotonic grid would place values wrongly."""
        for feature in service.feature_distribution()["features"].values():
            quantiles = feature["quantiles"]
            assert quantiles == sorted(quantiles)

    def test_the_grid_and_the_quantiles_line_up(self, service: PredictionService) -> None:
        distribution = service.feature_distribution()
        for feature in distribution["features"].values():
            assert len(feature["quantiles"]) == len(distribution["grid"])

    def test_the_grid_spans_zero_to_one(self, service: PredictionService) -> None:
        grid = service.feature_distribution()["grid"]
        assert grid[0] == 0.0
        assert grid[-1] == 1.0

    def test_it_is_cached(self, service: PredictionService) -> None:
        assert service.feature_distribution() is service.feature_distribution()

    def test_it_serialises_to_json(self, service: PredictionService) -> None:
        import json

        assert json.loads(json.dumps(service.feature_distribution()))["features"]


class TestCurrentSeasonPanel:
    """The season being played, served alongside the labelled ones.

    Its rows have every feature and no target. The risk is not that they fail
    to predict — that part is easy — but that a null target leaks into
    something that averages or compares values, where it would read as zero.
    """

    @pytest.fixture
    def with_current(self, artifact: ModelArtifact, players: pd.DataFrame) -> PredictionService:
        current = players[players["season"] == 2024].copy()
        current["season"] = 2025
        current[TARGET_COLUMN] = np.nan
        return PredictionService(
            {"performance_only": artifact},
            players,
            {pid: f"Player {pid}" for pid in players["player_id"].unique()},
            current_season=current,
        )

    def test_a_prediction_uses_the_freshest_season(self, with_current: PredictionService) -> None:
        """The point of the whole exercise: the default answer is about the
        season being played, not the last one that happened to be labelled."""
        assert with_current.predict_for_player(1).season == 2025

    def test_the_prediction_is_a_real_number(self, with_current: PredictionService) -> None:
        assert with_current.predict_for_player(1).prediction_eur > 0

    def test_the_current_season_appears_with_a_null_value_not_a_zero(
        self, with_current: PredictionService
    ) -> None:
        seasons = {row["season"]: row for row in with_current.player(1)["seasons"]}
        assert seasons[2025]["market_value_in_eur"] is None
        assert seasons[2025]["has_label"] is False
        assert seasons[2024]["has_label"] is True

    def test_history_excludes_the_unlabelled_season(self, with_current: PredictionService) -> None:
        """Every point carries `actual_eur`. A season with no actual has no
        point to make, and a zero would read as a collapse in value."""
        seasons = [point["season"] for point in with_current.prediction_history(1)]
        assert 2025 not in seasons
        assert 2024 in seasons

    def test_neighbours_are_never_unlabelled_rows(self, with_current: PredictionService) -> None:
        """A neighbour is displayed with its market value."""
        for neighbour in with_current.similar_players(1):
            assert neighbour["market_value_in_eur"] is not None

    def test_the_distribution_is_computed_over_recorded_values_only(
        self, with_current: PredictionService, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """The current-season rows here are a copy of 2024 with the value
        removed. Including them would shift every percentile toward the 2024
        distribution; excluding them leaves the panel exactly as it was."""
        labelled_only = PredictionService({"performance_only": artifact}, players, None)
        assert with_current.feature_distribution() == labelled_only.feature_distribution()

    def test_the_unlabelled_rows_are_not_silently_dropped_from_prediction(
        self, with_current: PredictionService, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """Excluded from the value summaries, included where they belong."""
        labelled_only = PredictionService({"performance_only": artifact}, players, None)
        assert with_current.predict_for_player(1).season == 2025
        assert labelled_only.predict_for_player(1).season == 2024

    def test_a_service_without_a_current_table_behaves_exactly_as_before(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """The table is optional: a deployment built before it existed serves."""
        service = PredictionService({"performance_only": artifact}, players, None)
        assert service.predict_for_player(1).season == 2024
        assert service.prediction_history(1)


class TestTheDisplayGateRatherThanEveryFeature:
    """Two bugs the final audit found, both silent, both about which rows the
    service is willing to show.

    Neither produced an error. Both produced an empty list, which reads to a
    caller exactly like "there is nothing to show" rather than "this endpoint
    is broken".
    """

    def test_a_row_with_a_null_feature_is_still_shown(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """The pipeline imputes. Requiring every feature to be non-null threw
        away 65.3% of the real panel — every player's first season among them —
        and left 14,402 of 24,411 players with an empty career chart.
        """
        holed = players.copy()
        holed.loc[holed.index[:5], "goals_per_90"] = np.nan
        service = PredictionService({"performance_only": artifact}, holed, NAMES)

        history = service.prediction_history(int(holed.iloc[0]["player_id"]))
        seasons_on_record = holed[holed["player_id"] == holed.iloc[0]["player_id"]]
        assert len(history) == len(seasons_on_record)

    def test_a_row_with_no_target_is_not_shown(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """The gate is the target, not the features: a history point carries an
        actual and a neighbour is displayed with its market value."""
        current = players[players["season"] == 2024].copy()
        current["season"] = 2025
        current[TARGET_COLUMN] = np.nan
        service = PredictionService(
            {"performance_only": artifact}, players, NAMES, current_season=current
        )
        assert all(point["season"] != 2025 for point in service.prediction_history(1))

    def test_comparables_anchor_on_the_latest_labelled_season(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """Anchoring on the latest *row* broke this for every active player.

        The neighbour pool is labelled-only, so a player whose most recent row
        is the season being played got an empty pool and an empty list — 8,709
        players, 32.8% of the panel, and exactly the ones anyone searches for.
        """
        current = players[players["season"] == 2024].copy()
        current["season"] = 2025
        current[TARGET_COLUMN] = np.nan
        service = PredictionService(
            {"performance_only": artifact}, players, NAMES, current_season=current
        )
        neighbours = service.similar_players(1)
        assert neighbours, "an active player must still get comparables"
        assert all(row["season"] == 2024 for row in neighbours)

    def test_a_player_with_no_labelled_season_gets_nothing_rather_than_a_guess(
        self, artifact: ModelArtifact, players: pd.DataFrame
    ) -> None:
        """The honest end of the same rule. A debutant in the season being
        played has no valued season to compare, and an empty list is the right
        answer — not a neighbour from a season he did not play in."""
        current = players[players["season"] == 2024].head(1).copy()
        current["season"] = 2025
        current["player_id"] = 9999
        current[TARGET_COLUMN] = np.nan
        service = PredictionService(
            {"performance_only": artifact}, players, NAMES, current_season=current
        )
        assert service.similar_players(9999) == []


class TestImplausibleValuesAreRejected:
    """`data.min_age: 15` and `data.max_age: 45` sat in `configs/config.yaml`,
    parsed and cross-validated, and read by nothing. The API answered for a
    1,000,000,000,000-year-old with a confident number and a 200 — the exact
    failure mode `NON_NEGATIVE_FEATURES` exists to prevent, one field over.
    """

    def test_an_impossible_age_is_a_bad_request(self, service: PredictionService) -> None:
        with pytest.raises(InvalidFeaturesError) as caught:
            service.predict_from_features({"age": 1e12})
        assert "age" in caught.value.message

    def test_a_too_young_age_is_a_bad_request(self, service: PredictionService) -> None:
        with pytest.raises(InvalidFeaturesError):
            service.predict_from_features({"age": 3.0})

    def test_an_age_inside_the_range_still_predicts(self, service: PredictionService) -> None:
        assert service.predict_from_features({"age": 24.0}).prediction_eur > 0

    def test_the_bounds_are_inclusive(self, service: PredictionService) -> None:
        """15 and 45 are ages a player can be, not ones he cannot."""
        assert service.predict_from_features({"age": 15.0}).prediction_eur > 0
        assert service.predict_from_features({"age": 45.0}).prediction_eur > 0


# --------------------------------------------------------------------------
# F7 — the prior-value variant must be handed its prior value
# --------------------------------------------------------------------------

PRIOR_FEATURES = FEATURES + ("prev_log_market_value_in_eur", "prev_value_age_days")


@pytest.fixture(scope="module")
def priced_players(players: pd.DataFrame) -> pd.DataFrame:
    """The panel with a prior valuation on every row except player 3's latest."""
    out = players.assign(
        prev_log_market_value_in_eur=np.log1p(players[TARGET_COLUMN] * 0.8),
        prev_value_age_days=60.0,
    )
    gap = (out["player_id"] == 3) & (out["season"] == 2024)
    out.loc[gap, ["prev_log_market_value_in_eur", "prev_value_age_days"]] = np.nan
    return out


@pytest.fixture(scope="module")
def prior_artifact(priced_players: pd.DataFrame) -> ModelArtifact:
    rows = priced_players.dropna(subset=["prev_log_market_value_in_eur"])
    numeric = [*NUMERIC, "prev_log_market_value_in_eur", "prev_value_age_days"]
    pipeline = build_pipeline(MODEL_REGISTRY["lightgbm"], numeric, CATEGORICAL)
    pipeline.fit(rows[list(PRIOR_FEATURES)], rows[TARGET_COLUMN])
    metrics = evaluate(rows[TARGET_COLUMN], pipeline.predict(rows[list(PRIOR_FEATURES)]))
    return ModelArtifact(
        variant="with_prior_value",
        model_name="lightgbm",
        params={},
        pipeline=pipeline,
        validation=metrics,
        test=metrics,
        feature_columns=PRIOR_FEATURES,
        target_column=TARGET_COLUMN,
        feature_importance=extract_feature_importance(pipeline),
        dataset={"rows": len(rows)},
        split={"strategy": "temporal"},
        seed=42,
    )


@pytest.fixture
def two_variants(
    artifact: ModelArtifact, prior_artifact: ModelArtifact, priced_players: pd.DataFrame
) -> PredictionService:
    return PredictionService(
        {"performance_only": artifact, "with_prior_value": prior_artifact}, priced_players, NAMES
    )


class TestPriorValueInputs:
    @pytest.mark.parametrize(
        ("features", "fragment"),
        [
            ({"prev_value_age_days": -5}, "cannot be negative"),
            ({"prev_value_age_days": 10**400}, "too large"),
            ({"prev_value_age_days": 30.5}, "whole number"),
            # A raw euro amount where a log one belongs: 1e308 and 500 were 200s.
            ({"prev_log_market_value_in_eur": 500.0}, "expected 0-23.03"),
            ({"prev_log_market_value_in_eur": 1e308}, "expected 0-23.03"),
        ],
    )
    def test_an_impossible_prior_is_a_bad_request(
        self, two_variants: PredictionService, features: dict[str, object], fragment: str
    ) -> None:
        seed = {"prev_log_market_value_in_eur": 14.0, "prev_value_age_days": 30}
        with pytest.raises(InvalidFeaturesError) as caught:
            two_variants.predict_from_features({**seed, **features}, variant="with_prior_value")
        assert fragment in caught.value.message
        assert next(iter(features)) in caught.value.message

    def test_the_bounds_of_a_prior_are_accepted(self, two_variants: PredictionService) -> None:
        for log_value in (0.0, 23.03):
            result = two_variants.predict_from_features(
                {"prev_log_market_value_in_eur": log_value, "prev_value_age_days": 0},
                variant="with_prior_value",
            )
            assert result.prediction_eur > 0

    """F7. A what-if request seeded from the default (performance-only) feature
    set and sent to the prior-value model carried no prior valuation; the
    pipeline imputed the training median and answered with a 200."""

    def test_omitting_the_prior_value_is_rejected_not_imputed(
        self, two_variants: PredictionService
    ) -> None:
        seed = two_variants.features_for(1, variant="performance_only")
        with pytest.raises(InvalidFeaturesError) as caught:
            two_variants.predict_from_features(seed, variant="with_prior_value")
        assert caught.value.code == "validation_error"
        assert set(caught.value.detail) == {"prev_log_market_value_in_eur", "prev_value_age_days"}

    def test_an_explicit_null_prior_is_rejected_too(self, two_variants: PredictionService) -> None:
        seed = two_variants.features_for(1, variant="with_prior_value")
        seed["prev_value_age_days"] = None
        with pytest.raises(InvalidFeaturesError):
            two_variants.predict_from_features(seed, variant="with_prior_value")

    def test_a_stored_row_with_no_prior_is_refused_by_that_variant_only(
        self, two_variants: PredictionService
    ) -> None:
        """A NaN prior passed the old `is None` check and was imputed."""
        with pytest.raises(InvalidFeaturesError):
            two_variants.predict_for_player(3, variant="with_prior_value")
        assert two_variants.predict_for_player(3, variant="performance_only").prediction_eur > 0

    def test_the_player_payload_seeds_the_requested_variant(
        self, two_variants: PredictionService
    ) -> None:
        """The fix for the what-if panel: it asks for the variant it will send to."""
        features = two_variants.player(1, variant="with_prior_value")["features"]
        assert set(features) == set(PRIOR_FEATURES)
        assert set(two_variants.player(1)["features"]) == set(FEATURES)

    def test_a_seed_for_the_variant_reproduces_its_stored_prediction(
        self, two_variants: PredictionService
    ) -> None:
        features = two_variants.player(1, variant="with_prior_value")["features"]
        stored = two_variants.predict_for_player(1, variant="with_prior_value")
        replayed = two_variants.predict_from_features(features, variant="with_prior_value")
        assert replayed.prediction_eur == pytest.approx(stored.prediction_eur)
