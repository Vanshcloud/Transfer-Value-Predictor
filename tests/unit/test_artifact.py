"""The saved artifact, and the one guarantee that makes it worth saving.

A model file that cannot reproduce the numbers recorded beside it is worse than
no file: it looks authoritative and is not. The central test here saves, loads,
re-predicts and compares against the recorded metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate
from src.feature_engineering.build import TARGET_COLUMN
from src.models import provenance
from src.models.artifact import (
    ARTIFACT_VERSION,
    MANIFEST_NAME,
    ModelArtifact,
    extract_feature_importance,
    load,
    load_manifest,
    load_variant,
    predict,
    save,
)
from src.models.registry import MODEL_REGISTRY, build_pipeline

NUMERIC = ["age", "goals"]
CATEGORICAL = ["position"]
FEATURES = ("age", "goals", "position")


def reload(path: Path) -> ModelArtifact:
    return load(path, sha256=provenance.sha256_file(path))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    size = 200
    goals = rng.integers(0, 20, size)
    return pd.DataFrame(
        {
            "age": rng.uniform(18, 34, size),
            "goals": goals,
            "position": rng.choice(["Attack", "Defender"], size),
            TARGET_COLUMN: np.expm1(13 + 0.1 * goals + rng.normal(0, 0.2, size)),
        }
    )


def make_artifact(frame: pd.DataFrame, model_name: str = "ridge") -> ModelArtifact:
    pipeline = build_pipeline(MODEL_REGISTRY[model_name], NUMERIC, CATEGORICAL)
    pipeline.fit(frame[list(FEATURES)], frame[TARGET_COLUMN])
    metrics = evaluate(frame[TARGET_COLUMN], pipeline.predict(frame[list(FEATURES)]))

    return ModelArtifact(
        variant="performance_only",
        model_name=model_name,
        params={"model__regressor__alpha": 1.0},
        pipeline=pipeline,
        validation=metrics,
        test=metrics,
        feature_columns=FEATURES,
        target_column=TARGET_COLUMN,
        feature_importance=extract_feature_importance(pipeline),
        dataset={"table": "training_table", "rows": len(frame)},
        split={"strategy": "temporal"},
        seed=42,
        leaderboard=[{"model": model_name, "validation_mae_eur": metrics.mae}],
    )


class TestRoundTrip:
    def test_a_reloaded_artifact_reproduces_its_recorded_metrics_exactly(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        """The guarantee everything else rests on.

        Phase 7's verification, and the reason the fitted preprocessing lives
        inside the pipeline rather than beside it.
        """
        artifact = make_artifact(frame)
        reloaded = reload(save(artifact, tmp_path))

        recomputed = evaluate(frame[TARGET_COLUMN], predict(reloaded, frame))
        assert recomputed == artifact.test

    def test_predictions_survive_the_round_trip_bit_for_bit(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        artifact = make_artifact(frame)
        before = predict(artifact, frame)
        after = predict(reload(save(artifact, tmp_path)), frame)
        np.testing.assert_array_equal(before, after)

    def test_metadata_survives(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        reloaded = reload(save(make_artifact(frame), tmp_path))
        assert reloaded.variant == "performance_only"
        assert reloaded.feature_columns == FEATURES
        assert reloaded.seed == 42
        assert reloaded.split["strategy"] == "temporal"

    def test_column_order_does_not_change_predictions(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        # predict() selects by the artifact's own column list, so a caller
        # handing over a differently ordered frame cannot silently shift inputs.
        artifact = reload(save(make_artifact(frame), tmp_path))
        shuffled = frame[list(reversed(list(frame.columns)))]
        np.testing.assert_array_equal(predict(artifact, frame), predict(artifact, shuffled))


class TestSidecar:
    def test_a_readable_json_lands_beside_the_model(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        # "Which run was best?" should not require unpickling nine models.
        path = save(make_artifact(frame), tmp_path)
        sidecar = path.with_suffix(".json")
        assert sidecar.exists()

        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        assert metadata["model_name"] == "ridge"
        assert metadata["metrics"]["test"]["mae_eur"] > 0
        assert metadata["seed"] == 42

    def test_the_sidecar_records_every_tracked_field(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        """The contract in docs/EXPERIMENT_TRACKING.md, asserted."""
        metadata = json.loads(
            save(make_artifact(frame), tmp_path).with_suffix(".json").read_text(encoding="utf-8")
        )
        assert {
            "artifact_version",
            "created_at",
            "variant",
            "model_name",
            "params",
            "metrics",
            "feature_columns",
            "target_column",
            "feature_importance",
            "dataset",
            "split",
            "seed",
            "leaderboard",
        } <= set(metadata)

    def test_the_file_name_identifies_variant_and_model(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = save(make_artifact(frame), tmp_path)
        assert path.name == "performance_only__ridge.joblib"


class TestVersioning:
    def test_an_artifact_from_a_future_schema_is_refused(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        # An old artifact read by new code must fail loudly, not be misread.
        artifact = make_artifact(frame)
        artifact.artifact_version = ARTIFACT_VERSION + 1
        path = save(artifact, tmp_path)

        with pytest.raises(ValueError, match="artifact_version"):
            reload(path)


class TestManifest:
    """F19: which file serves a variant is recorded, never inferred from names."""

    def test_saving_records_the_file_and_its_checksum(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = save(make_artifact(frame), tmp_path)
        entry = load_manifest(tmp_path)["performance_only"]  # type: ignore[index]
        assert entry["file"] == path.name
        assert entry["sha256"] == provenance.sha256_file(path)
        sidecar = json.loads(path.with_suffix(".json").read_text())
        assert sidecar["sha256"] == entry["sha256"]

    def test_the_latest_save_of_a_variant_is_the_one_recorded(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        save(make_artifact(frame, "lightgbm"), tmp_path)
        save(make_artifact(frame, "ridge"), tmp_path)
        assert load_manifest(tmp_path)["performance_only"]["file"] == "performance_only__ridge.joblib"  # type: ignore[index]

    def test_a_tampered_file_is_refused_before_it_is_unpickled(
        self, frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = save(make_artifact(frame), tmp_path)
        expected = load_manifest(tmp_path)["performance_only"]["sha256"]  # type: ignore[index]
        path.write_bytes(path.read_bytes() + b"\0")

        import src.models.artifact as module

        def must_not_run(*_: object) -> None:
            raise AssertionError("unpickled a file whose checksum did not match")

        monkeypatch.setattr(module.joblib, "load", must_not_run)
        with pytest.raises(ValueError, match="checksum"):
            load(path, sha256=expected)

    @pytest.mark.parametrize(
        "content",
        [
            "not json",
            '{"variants": {"performance_only": {"file": "x.joblib"}}}',
            '{"variants": {"performance_only": {"file": "../elsewhere.joblib", "sha256": "0"}}}',
        ],
    )
    def test_a_malformed_manifest_is_an_error_not_an_empty_one(
        self, tmp_path: Path, content: str
    ) -> None:
        (tmp_path / MANIFEST_NAME).write_text(content)
        with pytest.raises(ValueError):
            load_manifest(tmp_path)

    def test_no_manifest_is_none(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path) is None


GOOD_HASH = "0" * 64


class TestIntegrity:
    """Re-audit: no manifest content may route a file around verification."""

    @pytest.fixture(autouse=True)
    def never_unpickle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.models.artifact as module

        def must_not_run(*_: object) -> None:
            raise AssertionError("unpickled a file that was never verified")

        monkeypatch.setattr(module.joblib, "load", must_not_run)

    @pytest.mark.parametrize(
        "sha256", [None, "", 12345, ["a" * 64], "abc", "A" * 64, "g" * 64, "0" * 65]
    )
    def test_a_missing_null_or_malformed_hash_invalidates_the_manifest(
        self, tmp_path: Path, sha256: object
    ) -> None:
        entry: dict[str, object] = {"file": "performance_only__ridge.joblib", "sha256": sha256}
        (tmp_path / MANIFEST_NAME).write_text(
            json.dumps({"variants": {"performance_only": entry}})
        )
        with pytest.raises(ValueError, match="sha256"):
            load_manifest(tmp_path)

    @pytest.mark.parametrize(
        "content",
        [
            "[]",
            '{"variants": null}',
            '{"variants": []}',
            '{"variants": {"performance_only": null}}',
            '{"variants": {"performance_only": "x.joblib"}}',
            f'{{"variants": {{"performance_only": {{"file": 7, "sha256": "{GOOD_HASH}"}}}}}}',
            f'{{"variants": {{"performance_only": {{"file": null, "sha256": "{GOOD_HASH}"}}}}}}',
            f'{{"variants": {{"performance_only": {{"file": "", "sha256": "{GOOD_HASH}"}}}}}}',
            f'{{"variants": {{"performance_only": {{"sha256": "{GOOD_HASH}"}}}}}}',
        ],
    )
    def test_a_malformed_entry_is_a_controlled_error(self, tmp_path: Path, content: str) -> None:
        (tmp_path / MANIFEST_NAME).write_text(content)
        with pytest.raises(ValueError):
            load_manifest(tmp_path)

    def test_a_manifest_that_is_not_utf8_is_a_controlled_error(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_NAME).write_bytes(b"\xff\xfe\x00garbage")
        with pytest.raises(ValueError):
            load_manifest(tmp_path)

    @pytest.mark.parametrize("sha256", [None, "", "abc", 0])
    def test_load_refuses_to_skip_verification(self, tmp_path: Path, sha256: object) -> None:
        path = tmp_path / "performance_only__ridge.joblib"
        path.write_bytes(b"anything")
        with pytest.raises(ValueError, match="sha256"):
            load(path, sha256=sha256)  # type: ignore[arg-type]

    def test_a_corrupted_file_is_refused_before_it_is_unpickled(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = save(make_artifact(frame), tmp_path)
        path.write_bytes(b"\x80\x04not a pickle at all")
        with pytest.raises(ValueError, match="checksum"):
            load_variant(tmp_path, "performance_only")


class TestWhatTheFileHolds:
    """Checks that need the file unpickled, after its checksum has passed."""

    def test_a_misleading_file_name_does_not_decide_the_variant(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        # A performance_only model saved under the with_prior_value name, and
        # the manifest pointed at it with a correct checksum.
        real = save(make_artifact(frame), tmp_path)
        disguised = real.rename(tmp_path / "with_prior_value__lightgbm.joblib")
        (tmp_path / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "variants": {
                        "with_prior_value": {
                            "file": disguised.name,
                            "sha256": provenance.sha256_file(disguised),
                            "model_name": "lightgbm",
                        }
                    }
                }
            )
        )
        with pytest.raises(ValueError, match="performance_only model"):
            load_variant(tmp_path, "with_prior_value")

    def test_a_file_holding_another_family_than_its_entry_is_refused(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        save(make_artifact(frame, "ridge"), tmp_path)
        manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
        manifest["variants"]["performance_only"]["model_name"] = "lightgbm"
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="holds ridge"):
            load_variant(tmp_path, "performance_only")

    def test_a_verified_file_that_is_not_an_artifact_is_refused(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "performance_only__ridge.joblib"
        joblib.dump({"not": "an artifact"}, path)
        with pytest.raises(ValueError, match="not a ModelArtifact"):
            load(path, sha256=provenance.sha256_file(path))


class TestProvenance:
    """F10: an artifact says what it was made from."""

    def test_it_survives_the_round_trip(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        artifact = make_artifact(frame)
        artifact.provenance = {"source_revision": {"commit": "abc", "dirty": False}}
        assert reload(save(artifact, tmp_path)).provenance == artifact.provenance
        sidecar = json.loads((tmp_path / "performance_only__ridge.json").read_text())
        assert sidecar["provenance"]["source_revision"]["commit"] == "abc"

    def test_collect_hashes_each_input(self, tmp_path: Path) -> None:
        raw = tmp_path / "players.csv"
        raw.write_text("player_id\n1\n")
        table = tmp_path / "training_table.parquet"
        table.write_bytes(b"PAR1")
        config = tmp_path / "config.yaml"
        config.write_text("a: 1\n")

        record = provenance.collect(
            training_table=table,
            raw_files=[raw, tmp_path / "absent.csv"],
            dataset="owner/dataset",
            configuration={"data": {"label_tolerance_days": 365}},
            config_file=config,
        )
        assert record["raw_data"]["files"] == {
            "players.csv": {"sha256": provenance.sha256_file(raw), "bytes": raw.stat().st_size}
        }
        assert record["raw_data"]["reproducible_by_download"] is False
        assert record["training_table"]["sha256"] == provenance.sha256_file(table)
        assert record["configuration"]["sha256"] == provenance.sha256_file(config)
        assert record["environment"]["packages"]["scikit-learn"]

    def test_an_unavailable_git_is_recorded_rather_than_guessed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_git(*_: object, **__: object) -> None:
            raise OSError("git: not found")

        monkeypatch.setattr(provenance.subprocess, "run", no_git)
        revision = provenance.source_revision()
        assert revision["commit"] is None
        assert "not found" in revision["unavailable_because"]


class TestFeatureImportance:
    @pytest.mark.parametrize("model_name", ["ridge", "random_forest", "lightgbm"])
    def test_importances_are_named_after_preprocessing(
        self, model_name: str, frame: pd.DataFrame
    ) -> None:
        # Names come from the preprocessor, which is why it is configured with
        # set_output(transform="pandas"). Without it Phase 8's SHAP plots are a
        # wall of x0, x1, x2.
        artifact = make_artifact(frame, model_name)
        assert artifact.feature_importance
        assert any("goals" in name for name in artifact.feature_importance)

    def test_a_family_exposing_neither_coefficients_nor_importances_is_not_fatal(
        self, frame: pd.DataFrame
    ) -> None:
        pipeline = build_pipeline(MODEL_REGISTRY["ridge"], NUMERIC, CATEGORICAL)
        pipeline.fit(frame[list(FEATURES)], frame[TARGET_COLUMN])
        del pipeline.named_steps["model"].regressor_.coef_

        assert extract_feature_importance(pipeline) == {}
