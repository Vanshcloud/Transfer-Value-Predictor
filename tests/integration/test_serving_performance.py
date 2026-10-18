"""The shipped models meet the requirements they were selected under.

The selection policy (src/pipelines/tune._select_winner) admits only families
that explain a single prediction within SERVING_LATENCY_BUDGET_MS, and breaks
accuracy ties on the whole deployed footprint. These tests hold the *shipped*
artifacts to the same requirements and recompute the choice from what the
artifact recorded, so neither the budget nor the record can drift silently.

Marked ``integration``: needs the trained models and the training table.
"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import Metrics
from src.explainability.shap_explainer import explained_prediction_latency_ms, supports_shap
from src.feature_engineering.build import select_variant
from src.models.artifact import ModelArtifact, load_manifest, load_variant
from src.models.tuning import TuningResult
from src.pipelines.features import TRAINING_TABLE
from src.pipelines.train import VARIANTS
from src.pipelines.tune import SERVING_LATENCY_BUDGET_MS, FamilyResult, _select_winner
from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration

MODEL_DIR = PROJECT_ROOT / "models"


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def artifact(request: pytest.FixtureRequest) -> ModelArtifact:
    """One per variant, always — not one per file found, so the suite collects
    the same tests on a clean clone as on a machine with models."""
    if request.param not in (load_manifest(MODEL_DIR) or {}):
        pytest.skip(f"no {request.param} model; run scripts/train_models.py")
    return load_variant(MODEL_DIR, request.param)


def test_every_artifact_on_disk_is_the_one_its_manifest_serves() -> None:
    """A stray second artifact is ignored by the service, but it is still a
    file someone will open believing it is the model."""
    manifest = load_manifest(MODEL_DIR)
    if manifest is None and not any(MODEL_DIR.glob("*.joblib")):
        pytest.skip("no trained models; run scripts/train_models.py")
    listed = {entry["file"] for entry in (manifest or {}).values()}
    assert {path.name for path in MODEL_DIR.glob("*.joblib")} == listed


def test_the_shipped_model_explains_a_prediction(artifact: ModelArtifact) -> None:
    assert supports_shap(artifact)


def test_the_shipped_model_meets_the_latency_budget(artifact: ModelArtifact) -> None:
    """Measured on the validation season, the rows selection measured on."""
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    if not store.has_table(TRAINING_TABLE):
        pytest.skip("run scripts/build_features.py first")
    frame, _ = select_variant(
        store.read_table(TRAINING_TABLE),
        include_prior_value=artifact.variant == "with_prior_value",
    )
    validation = frame[frame["season"] == artifact.split["validation_season"]]
    latency = explained_prediction_latency_ms(artifact, validation)
    assert latency is not None
    assert latency <= SERVING_LATENCY_BUDGET_MS, f"{artifact.variant}: {latency:.1f} ms"


def _from_record(row: dict[str, object]) -> FamilyResult:
    metrics = Metrics(
        mae=float(row["validation_mae_eur"]),  # type: ignore[arg-type]
        rmse=float(row["validation_rmse_eur"]),  # type: ignore[arg-type]
        r2=float(row["validation_r2"]),  # type: ignore[arg-type]
        mape=float(row["validation_mape"]),  # type: ignore[arg-type]
        n=int(row["predict_rows"]),  # type: ignore[call-overload]
        mae_standard_error=float(row["validation_mae_standard_error"]),  # type: ignore[arg-type]
    )
    return FamilyResult(
        tuning=TuningResult(
            model_name=str(row["model"]),
            best_params=dict(row["params"]),  # type: ignore[call-overload]
            cv_mae=float(row["cv_mae_eur"]),  # type: ignore[arg-type]
            n_candidates=1,
        ),
        validation=metrics,
        fit_seconds=float(row["fit_seconds"]),  # type: ignore[arg-type]
        predict_seconds=float(row["predict_seconds"]),  # type: ignore[arg-type]
        size_bytes=int(row["size_bytes"]),  # type: ignore[call-overload]
        explained_latency_ms=(
            None
            if row["explained_latency_ms"] is None
            else float(row["explained_latency_ms"])  # type: ignore[arg-type]
        ),
    )


def test_the_choice_is_recomputable_from_the_artifact(artifact: ModelArtifact) -> None:
    """Every input the policy reads is in the leaderboard, and re-applying the
    policy to it names the shipped family. No test metric is involved."""
    results = [_from_record(row) for row in artifact.leaderboard]
    assert _select_winner(results, artifact.variant).name == artifact.model_name


def test_every_excluded_family_says_why(artifact: ModelArtifact) -> None:
    winner = next(row for row in artifact.leaderboard if row["model"] == artifact.model_name)
    assert winner["ineligible_because"] is None
    for row in artifact.leaderboard:
        reason = row["ineligible_because"]
        assert reason is None or isinstance(reason, str)
