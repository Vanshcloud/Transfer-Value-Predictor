"""Model card, comparison table and HTML reports.

These are the outputs a person reads and then acts on, so the tests are about
honesty as much as correctness: that the card states its limitations, that the
comparison shows cost as well as accuracy, and that a report is genuinely one
self-contained file rather than one that renders blank without a network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison import (
    DISPLAY_NAMES,
    build_comparison_document,
    comparison_frame,
)
from src.evaluation.error_analysis import analyse_errors
from src.evaluation.metrics import evaluate
from src.evaluation.model_card import build_model_card
from src.evaluation.reports import (
    error_analysis_report,
    evaluation_report,
    feature_importance_report,
)
from src.feature_engineering.build import TARGET_COLUMN
from src.models.artifact import ModelArtifact
from src.models.registry import MODEL_REGISTRY, build_pipeline
from src.visualization import plots

NUMERIC = ["age", "goals"]
CATEGORICAL = ["position"]
FEATURES = ("age", "goals", "position")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    size = 250
    goals = rng.integers(0, 25, size)
    return pd.DataFrame(
        {
            "player_id": range(size),
            "season": rng.choice([2023, 2024], size),
            "age": rng.uniform(18, 35, size),
            "goals": goals,
            "position": rng.choice(["Attack", "Defender"], size),
            TARGET_COLUMN: np.expm1(13 + 0.1 * goals + rng.normal(0, 0.3, size)),
        }
    )


@pytest.fixture(scope="module")
def artifact(frame: pd.DataFrame) -> ModelArtifact:
    pipeline = build_pipeline(MODEL_REGISTRY["lightgbm"], NUMERIC, CATEGORICAL)
    pipeline.fit(frame[list(FEATURES)], frame[TARGET_COLUMN])
    metrics = evaluate(frame[TARGET_COLUMN], pipeline.predict(frame[list(FEATURES)]))

    return ModelArtifact(
        variant="with_prior_value",
        model_name="lightgbm",
        params={"model__regressor__num_leaves": 31},
        pipeline=pipeline,
        validation=metrics,
        test=metrics,
        feature_columns=FEATURES,
        target_column=TARGET_COLUMN,
        feature_importance={"numeric__goals": 120.0, "numeric__age": 80.0},
        dataset={"rows": len(frame)},
        split={
            "strategy": "temporal",
            "train_end_season": 2021,
            "validation_season": 2022,
            "test_start_season": 2023,
        },
        seed=42,
        leaderboard=[
            {
                "model": "lightgbm",
                "validation_mae_eur": 2_988_615.0,
                "validation_rmse_eur": 7_153_067.0,
                "validation_r2": 0.72,
                "validation_mape": 0.48,
                "cv_mae_eur": 2_700_765.0,
                "fit_seconds": 0.81,
                "predict_seconds": 0.01,
                "predict_rows": 1569,
                "size_bytes": 943_635,
                "params": {},
            },
            {
                "model": "random_forest",
                "validation_mae_eur": 3_052_311.0,
                "validation_rmse_eur": 7_400_000.0,
                "validation_r2": 0.721,
                "validation_mape": 0.49,
                "cv_mae_eur": 2_757_024.0,
                "fit_seconds": 23.2,
                "predict_seconds": 0.09,
                "predict_rows": 1569,
                "size_bytes": 41_000_000,
                "params": {},
            },
        ],
    )


@pytest.fixture(scope="module")
def analysis(frame: pd.DataFrame, artifact: ModelArtifact) -> object:
    predictions = artifact.pipeline.predict(frame[list(FEATURES)])
    return analyse_errors(frame, predictions, target_column=TARGET_COLUMN)


class TestComparison:
    def test_it_shows_cost_next_to_accuracy(self, artifact: ModelArtifact) -> None:
        """A family that ties on MAE but is 40x the size lost for a visible reason."""
        frame = comparison_frame(artifact)
        assert {"train (s)", "predict (ms/1k)", "size (MB)"} <= set(frame.columns)

        forest = frame[frame["model"] == "Random Forest"].iloc[0]
        winner = frame[frame["model"] == "LightGBM"].iloc[0]
        assert forest["size (MB)"] > winner["size (MB)"] * 10
        assert forest["train (s)"] > winner["train (s)"] * 10

    def test_it_marks_the_selected_family(self, artifact: ModelArtifact) -> None:
        frame = comparison_frame(artifact)
        selected = frame[frame["selected"] == "yes"]
        assert len(selected) == 1
        assert selected.iloc[0]["model"] == DISPLAY_NAMES["lightgbm"]

    def test_every_registry_family_has_a_display_name(self) -> None:
        assert set(DISPLAY_NAMES) == set(MODEL_REGISTRY)

    def test_the_document_explains_why_test_metrics_are_absent_per_family(
        self, artifact: ModelArtifact
    ) -> None:
        # Ranking nine families on test would make the test set a second
        # validation set. The document has to say so.
        document = build_comparison_document([artifact])
        assert "touched once" in document
        assert "validation MAE" in document

    def test_the_document_is_valid_markdown_table(self, artifact: ModelArtifact) -> None:
        document = build_comparison_document([artifact])
        assert "| model |" in document
        assert "LightGBM" in document


class TestModelCard:
    def test_it_states_limitations(self, artifact: ModelArtifact, analysis: object) -> None:
        """A card that only lists metrics is a scoreboard."""
        card = build_model_card(artifact, analysis=analysis)  # type: ignore[arg-type]
        assert "## Limitations" in card
        assert "Coverage begins in 2012" in card

    def test_it_says_what_the_model_is_not_for(self, artifact: ModelArtifact) -> None:
        card = build_model_card(artifact)
        assert "Not what it is for" in card
        assert "transfer fee" in card

    def test_the_prior_value_variant_declares_its_anchoring(self, artifact: ModelArtifact) -> None:
        assert "anchored to the previous valuation" in build_model_card(artifact)

    def test_it_reports_measured_metrics_not_placeholders(self, artifact: ModelArtifact) -> None:
        card = build_model_card(artifact)
        assert f"€{artifact.test.mae:,.0f}" in card
        assert f"{artifact.test.r2:.3f}" in card

    def test_it_records_the_leakage_controls(self, artifact: ModelArtifact) -> None:
        card = build_model_card(artifact)
        assert "Leakage controls" in card
        assert "contract_expiration_date" in card

    def test_it_names_the_weakest_segment_when_given_an_analysis(
        self, artifact: ModelArtifact, analysis: object
    ) -> None:
        card = build_model_card(artifact, analysis=analysis)  # type: ignore[arg-type]
        assert "Weakest measured segment" in card


class TestHtmlReports:
    def test_a_report_is_one_self_contained_file(
        self, artifact: ModelArtifact, analysis: object, tmp_path: object
    ) -> None:
        """No CDN, no sidecar images: a report that needs a network is blank
        in the meeting where it matters."""
        from pathlib import Path

        report = evaluation_report(artifact, analysis)  # type: ignore[arg-type]
        path = report.write(Path(str(tmp_path)))
        content = path.read_text(encoding="utf-8")

        assert content.startswith("<!doctype html>")
        assert "data:image/png;base64," in content
        assert "http://" not in content and "https://" not in content
        assert list(Path(str(tmp_path)).iterdir()) == [path]

    def test_the_evaluation_report_states_the_split_it_used(
        self, artifact: ModelArtifact, analysis: object
    ) -> None:
        body = evaluation_report(artifact, analysis).body  # type: ignore[arg-type]
        assert "test seasons" in body
        assert "never saw" in body

    def test_the_importance_report_distinguishes_importance_from_shap(
        self, artifact: ModelArtifact
    ) -> None:
        body = feature_importance_report(artifact).body
        assert "moved predictions" in body

    def test_the_error_report_breaks_error_down_by_segment(
        self, artifact: ModelArtifact, analysis: object
    ) -> None:
        body = error_analysis_report(artifact, analysis).body  # type: ignore[arg-type]
        assert "value-band" in body or "value band" in body.lower()
        assert "Largest misses" in body

    def test_report_content_is_escaped(self) -> None:
        from src.evaluation.reports import Report

        page = Report("x", "<script>alert(1)</script>", "<p>ok</p>").to_html()
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page


class TestPlots:
    def test_figures_encode_to_data_uris(self) -> None:
        figure = plots.feature_importance({"a": 1.0, "b": 2.0})
        assert plots.to_data_uri(figure).startswith("data:image/png;base64,")

    def test_closing_figures_keeps_memory_flat(self) -> None:
        # matplotlib keeps every unclosed figure alive and a report builds dozens.
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())
        for _ in range(5):
            plots.to_data_uri(plots.feature_importance({"a": 1.0}))
        assert len(plt.get_fignums()) == before

    def test_a_waterfall_renders_for_one_prediction(self) -> None:
        figure = plots.contribution_waterfall(
            ["age", "goals"], [0.4, -0.2], 2_000_000.0, 2_500_000.0
        )
        assert plots.to_data_uri(figure).startswith("data:image/png;base64,")
