"""The prospective test's two guarantees (audit F5): a freeze cannot be
altered unnoticed, and scoring reads no valuation the models could have seen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate
from src.evaluation.prospective import FREEZE_FILE, PREDICTIONS_FILE, freeze, score
from src.feature_engineering.build import TARGET_COLUMN
from src.models.artifact import ModelArtifact, extract_feature_importance
from src.models.calibration import calibrate
from src.models.registry import MODEL_REGISTRY, build_pipeline

FEATURES = ("age", "goals", "position")
CUTOFF = pd.Timestamp("2026-06-12")


@pytest.fixture(scope="module")
def artifact() -> ModelArtifact:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "age": rng.uniform(18, 34, 200),
            "goals": rng.integers(0, 20, 200),
            "position": rng.choice(["Attack", "Defender"], 200),
        }
    )
    frame[TARGET_COLUMN] = np.expm1(13 + 0.1 * frame["goals"] + rng.normal(0, 0.2, 200))
    pipeline = build_pipeline(MODEL_REGISTRY["ridge"], ["age", "goals"], ["position"])
    pipeline.fit(frame[list(FEATURES)], frame[TARGET_COLUMN])
    predicted = pipeline.predict(frame[list(FEATURES)])
    metrics = evaluate(frame[TARGET_COLUMN], predicted)
    return ModelArtifact(
        variant="performance_only",
        model_name="ridge",
        params={},
        pipeline=pipeline,
        validation=metrics,
        test=metrics,
        feature_columns=FEATURES,
        target_column=TARGET_COLUMN,
        feature_importance=extract_feature_importance(pipeline),
        dataset={},
        split={},
        seed=42,
        calibration=calibrate(frame[TARGET_COLUMN].to_numpy(float), predicted),
    )


def current() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "season": [2025, 2025, 2025],
            "as_of_date": ["2026-07-01"] * 3,
            "age": [24.0, 29.0, 21.0],
            "goals": [10, 2, 15],
            "position": ["Attack", "Defender", "Attack"],
        }
    )


@pytest.fixture
def frozen(artifact: ModelArtifact, tmp_path: Path) -> Path:
    freeze(
        {"performance_only": (artifact, "abc")},
        current(),
        cutoff=CUTOFF,
        tolerance_days=365,
        directory=tmp_path,
        inputs={"current_season": "def"},
    )
    return tmp_path


def test_a_freeze_records_its_hash_and_its_rule(frozen: Path) -> None:
    record = json.loads((frozen / FREEZE_FILE).read_text())
    assert record["cutoff"] == "2026-06-12"
    assert record["models"] == {"performance_only": "abc"}
    assert record["rows"] == {"performance_only": 3}
    assert len(record["predictions_sha256"]) == 64


def test_a_freeze_is_never_rewritten(artifact: ModelArtifact, frozen: Path) -> None:
    with pytest.raises(FileExistsError):
        freeze(
            {"performance_only": (artifact, "abc")},
            current(),
            cutoff=CUTOFF,
            tolerance_days=365,
            directory=frozen,
            inputs={},
        )


def test_rows_whose_labels_may_already_exist_are_refused(
    artifact: ModelArtifact, tmp_path: Path
) -> None:
    early = current().assign(as_of_date="2026-06-01")
    with pytest.raises(ValueError, match="not prospective"):
        freeze(
            {"performance_only": (artifact, "abc")},
            early,
            cutoff=CUTOFF,
            tolerance_days=365,
            directory=tmp_path,
            inputs={},
        )


def test_scoring_uses_only_valuations_after_the_cutoff(frozen: Path) -> None:
    valuations = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 3],
            "date": ["2026-05-01", "2026-08-01", "2026-06-01", "2026-09-01"],
            "market_value_in_eur": [9e9, 2e6, 9e9, 5e6],
        }
    )
    result = score(frozen, valuations)["variants"]["performance_only"]
    # Player 2's only valuation predates the cutoff: unseen-by-construction
    # scoring must not use it, so the row is pending, not labelled.
    assert result["labelled"] == 2
    assert result["pending"] == 1
    assert result["complete"] is False
    assert result["mae_eur"] < 9e9  # the pre-cutoff 9e9 never entered a label


def test_a_closed_window_with_no_valuation_is_never_labelled(frozen: Path) -> None:
    valuations = pd.DataFrame(
        {
            "player_id": [1, 3, 9],
            "date": ["2026-08-01", "2026-08-01", "2027-08-01"],
            "market_value_in_eur": [2e6, 5e6, 1e6],
        }
    )
    result = score(frozen, valuations)["variants"]["performance_only"]
    assert result["complete"] is True
    assert result["never_labelled"] == 1 and result["pending"] == 0


def test_altered_predictions_are_refused(frozen: Path) -> None:
    path = frozen / PREDICTIONS_FILE
    path.write_text(path.read_text().replace("2025", "2024", 1))
    with pytest.raises(ValueError, match="hash"):
        score(frozen, pd.DataFrame({"player_id": [], "date": [], "market_value_in_eur": []}))
