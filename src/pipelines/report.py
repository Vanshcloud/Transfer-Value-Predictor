"""Turn saved artifacts into the evaluation and explainability outputs.

Reads models, never trains them: reporting is separated from training so a
presentation can be regenerated in seconds without a twenty-minute zoo run, and
so the numbers in a report provably come from the model on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.comparison import build_comparison_document, comparison_frame
from src.evaluation.error_analysis import ErrorAnalysis, analyse_errors
from src.evaluation.model_card import build_model_card
from src.evaluation.reports import (
    Report,
    baseline_report,
    error_analysis_report,
    evaluation_report,
    feature_importance_report,
    shap_report,
)
from src.explainability.shap_explainer import (
    GlobalExplanation,
    PredictionExplanation,
    explain_global,
    explain_prediction,
    supports_shap,
)
from src.feature_engineering.build import TARGET_COLUMN, select_variant
from src.models.artifact import ModelArtifact, load, predict
from src.models.splits import temporal_split
from src.pipelines.features import TRAINING_TABLE
from src.pipelines.train import VARIANTS
from src.storage.base import TableStore
from src.utils.config import SplitConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReportBundle:
    """Everything produced for one model."""

    artifact: ModelArtifact
    analysis: ErrorAnalysis
    explanation: GlobalExplanation | None
    written: list[Path]


def _example_players(analysis: ErrorAnalysis, count: int = 3) -> list[tuple[str, pd.DataFrame]]:
    """A spread of players to explain, not three near-identical ones.

    The most expensive player the model saw, one it got badly wrong, and one it
    got right — so a reader sees the explanation working and failing rather
    than three flattering cases.
    """
    residuals = analysis.residuals
    chosen: list[tuple[str, pd.DataFrame]] = []

    most_valuable = residuals.nlargest(1, TARGET_COLUMN)
    chosen.append(("Highest-valued player in the test seasons", most_valuable))

    worst = analysis.worst_overpredictions.head(1)
    if not worst.empty:
        chosen.append(("Largest overvaluation", worst))

    closest = residuals.nsmallest(1, "absolute_error")
    chosen.append(("A near-exact prediction", closest))

    return chosen[:count]


def build_reports(
    store: TableStore,
    config: SplitConfig,
    model_directory: Path,
    report_directory: Path,
    docs_directory: Path,
) -> list[ReportBundle]:
    """Generate every report for every saved artifact.

    Raises:
        FileNotFoundError: if no artifact has been trained yet.
    """
    table = store.read_table(TRAINING_TABLE)
    # Created up front: the model cards are written per variant, before the
    # comparison document that used to be the first thing to create this.
    docs_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    bundles: list[ReportBundle] = []
    artifacts: list[ModelArtifact] = []

    for variant, include_prior in VARIANTS.items():
        matches = sorted(model_directory.glob(f"{variant}__*.joblib"))
        if not matches:
            raise FileNotFoundError(
                f"no artifact for {variant} in {model_directory} — "
                "run scripts/train_models.py first"
            )

        artifact = load(matches[0])
        artifacts.append(artifact)
        bundles.append(
            _build_for_artifact(
                artifact, table, config, include_prior, report_directory, docs_directory
            )
        )

    # One comparison document covering both variants, since the whole point is
    # to see the families side by side.
    comparison_path = docs_directory / "model_comparison.md"
    comparison_path.write_text(build_comparison_document(artifacts), encoding="utf-8")
    logger.info("wrote %s", comparison_path)

    report = baseline_report(comparison_frame(artifacts[0]), artifacts[0])
    bundles[0].written.append(report.write(report_directory))
    bundles[0].written.append(comparison_path)

    return bundles


def _build_for_artifact(
    artifact: ModelArtifact,
    table: pd.DataFrame,
    config: SplitConfig,
    include_prior: bool,
    report_directory: Path,
    docs_directory: Path,
) -> ReportBundle:
    frame, _ = select_variant(table, include_prior_value=include_prior)
    split = temporal_split(
        frame,
        train_end_season=config.train_end_season,
        validation_season=config.validation_season,
        test_start_season=config.test_start_season,
    )
    test_rows = frame.loc[split.test]
    predictions: np.ndarray = predict(artifact, test_rows)

    analysis = analyse_errors(test_rows, predictions, target_column=TARGET_COLUMN)
    logger.info(
        "%s: test MAE EUR %s over %d rows",
        artifact.variant,
        f"{analysis.overall.mae:,.0f}",
        analysis.overall.n,
    )

    written: list[Path] = []
    suffix = "" if artifact.variant == "performance_only" else f"__{artifact.variant}"

    explanation: GlobalExplanation | None = None
    if supports_shap(artifact):
        explanation = explain_global(artifact, test_rows)
        examples: list[tuple[str, PredictionExplanation]] = [
            (label, explain_prediction(artifact, rows))
            for label, rows in _example_players(analysis)
        ]
        report = shap_report(artifact, explanation, examples)
        written.append(_write(report, report_directory, suffix))
    else:
        logger.warning(
            "%s is not a tree model; skipping SHAP and reporting coefficients instead",
            artifact.model_name,
        )

    for report in (
        evaluation_report(artifact, analysis),
        feature_importance_report(artifact),
        error_analysis_report(artifact, analysis),
    ):
        written.append(_write(report, report_directory, suffix))

    card_path = docs_directory / f"MODEL_CARD_{artifact.variant}.md"
    card_path.write_text(build_model_card(artifact, analysis=analysis), encoding="utf-8")
    written.append(card_path)
    logger.info("wrote %s", card_path)

    return ReportBundle(artifact, analysis, explanation, written)


def _write(report: Report, directory: Path, suffix: str) -> Path:
    """Write a report, disambiguating the second variant by filename.

    Both variants produce a page called "evaluation"; without the suffix the
    second would silently overwrite the first.
    """
    if suffix:
        report = replace(
            report,
            name=f"{report.name}{suffix}",
            title=f"{report.title} — {suffix.strip('_')}",
        )
    return report.write(directory)
