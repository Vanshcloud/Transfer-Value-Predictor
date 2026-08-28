"""Train the zoo, select a winner, save it as one self-describing artifact.

The order here is the discipline, not an implementation detail:

1. Search each family's grid on **expanding-window folds inside the training
   seasons**. The validation season is not involved.
2. Fit each family's winning configuration on the full training seasons and
   score it **once** on the validation season. That number selects the family.
3. Score the single winner **once** on the test seasons. That number is
   reported and never optimised against.

Every step re-runs the leakage validator, because the cheapest moment to catch
a leak is before a metric built on it has been written down and believed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import Metrics, evaluate
from src.feature_engineering.build import CATEGORICAL_FEATURES, TARGET_COLUMN, select_variant
from src.models.artifact import ModelArtifact, extract_feature_importance, save
from src.models.registry import MODEL_REGISTRY, ModelSpec, build_pipeline
from src.models.splits import RANDOM_SEED, Split, temporal_split
from src.models.tuning import Fold, TuningResult, season_folds, tune
from src.pipelines.features import TRAINING_TABLE, leakage_validator
from src.pipelines.train import VARIANTS
from src.storage.base import TableStore
from src.utils.config import SplitConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FamilyResult:
    """One model family's tuned score on the validation season."""

    tuning: TuningResult
    validation: Metrics

    @property
    def name(self) -> str:
        return self.tuning.model_name


def _numeric_features(feature_columns: tuple[str, ...]) -> list[str]:
    return [column for column in feature_columns if column not in CATEGORICAL_FEATURES]


def train_variant(
    table: pd.DataFrame,
    variant: str,
    include_prior_value: bool,
    config: SplitConfig,
    *,
    model_names: tuple[str, ...] | None = None,
) -> ModelArtifact:
    """Tune every family for one variant and return the winner as an artifact.

    Raises:
        ValidationError: if the split leaks. No model is selected from a
            leaking split — a metric that looks good for the wrong reason is
            worse than no metric, because it gets believed.
    """
    frame, feature_columns = select_variant(table, include_prior_value=include_prior_value)
    numeric = _numeric_features(feature_columns)

    split = temporal_split(
        frame,
        train_end_season=config.train_end_season,
        validation_season=config.validation_season,
        test_start_season=config.test_start_season,
    )
    validator = leakage_validator(feature_columns)
    validator.raise_for_leakage(frame, splits=split.as_dict(), groups=frame["player_id"])

    folds = season_folds(frame, split.train)
    logger.info(
        "%s: %d rows, %d features, %d expanding-window fold(s)",
        variant,
        len(frame),
        len(feature_columns),
        len(folds),
    )

    names = model_names or tuple(MODEL_REGISTRY)
    results = [
        _tune_and_validate(MODEL_REGISTRY[name], frame, split, folds, feature_columns, numeric)
        for name in names
    ]

    # Selection is by validation MAE in EUR. RMSE would let a handful of
    # EUR 100M outliers choose the model; R2 is reported, never optimised.
    winner = min(results, key=lambda result: result.validation.mae)
    logger.info("%s: selected %s", variant, winner.name)

    return _fit_winner(winner, frame, split, feature_columns, numeric, variant, config, results)


def _tune_and_validate(
    spec: ModelSpec,
    frame: pd.DataFrame,
    split: Split,
    folds: list[Fold],
    feature_columns: tuple[str, ...],
    numeric: list[str],
) -> FamilyResult:
    tuning = tune(
        spec,
        frame,
        folds,
        feature_columns=feature_columns,
        numeric_features=numeric,
        categorical_features=CATEGORICAL_FEATURES,
        target_column=TARGET_COLUMN,
    )
    pipeline = build_pipeline(spec, numeric, CATEGORICAL_FEATURES)
    pipeline.set_params(**tuning.best_params)

    features = list(feature_columns)
    train, validation = frame.loc[split.train], frame.loc[split.validation]
    pipeline.fit(train[features], train[TARGET_COLUMN])

    return FamilyResult(
        tuning=tuning,
        validation=evaluate(validation[TARGET_COLUMN], pipeline.predict(validation[features])),
    )


def _fit_winner(
    winner: FamilyResult,
    frame: pd.DataFrame,
    split: Split,
    feature_columns: tuple[str, ...],
    numeric: list[str],
    variant: str,
    config: SplitConfig,
    results: list[FamilyResult],
) -> ModelArtifact:
    """Refit the winner and touch the test seasons, once."""
    pipeline = build_pipeline(MODEL_REGISTRY[winner.name], numeric, CATEGORICAL_FEATURES)
    pipeline.set_params(**winner.tuning.best_params)

    features = list(feature_columns)
    train = frame.loc[split.train]
    pipeline.fit(train[features], train[TARGET_COLUMN])

    test = frame.loc[split.test]
    test_metrics = evaluate(test[TARGET_COLUMN], pipeline.predict(test[features]))

    return ModelArtifact(
        variant=variant,
        model_name=winner.name,
        params=winner.tuning.best_params,
        pipeline=pipeline,
        validation=winner.validation,
        test=test_metrics,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        feature_importance=extract_feature_importance(pipeline),
        dataset={"table": TRAINING_TABLE, "rows": len(frame), "seasons": split.sizes},
        split={
            "strategy": "temporal",
            "train_end_season": config.train_end_season,
            "validation_season": config.validation_season,
            "test_start_season": config.test_start_season,
        },
        seed=RANDOM_SEED,
        leaderboard=[
            {
                "model": result.name,
                "validation_mae_eur": result.validation.mae,
                "validation_r2": result.validation.r2,
                "cv_mae_eur": result.tuning.cv_mae,
                "params": result.tuning.best_params,
            }
            for result in sorted(results, key=lambda r: r.validation.mae)
        ],
    )


def run_zoo(
    store: TableStore,
    config: SplitConfig,
    model_directory: Path,
    *,
    model_names: tuple[str, ...] | None = None,
) -> list[ModelArtifact]:
    """Train both variants and save one artifact each."""
    table = store.read_table(TRAINING_TABLE)

    artifacts = []
    for variant, include_prior in VARIANTS.items():
        artifact = train_variant(table, variant, include_prior, config, model_names=model_names)
        save(artifact, model_directory)
        artifacts.append(artifact)
    return artifacts


def render_leaderboard(artifact: ModelArtifact) -> str:
    """Every family's validation score, so "which was best?" stays answerable."""
    header = f"{'model':<18} {'val MAE EUR':>14} {'val R2':>8} {'cv MAE EUR':>14}"
    lines = [f"{artifact.variant}", header, "-" * len(header)]

    for row in artifact.leaderboard:
        marker = " <-- selected" if row["model"] == artifact.model_name else ""
        cv = row["cv_mae_eur"]
        cv_text = "n/a" if cv != cv else f"{cv:,.0f}"  # NaN when no folds ran
        lines.append(
            f"{row['model']:<18} {row['validation_mae_eur']:>14,.0f} "
            f"{row['validation_r2']:>8.3f} {cv_text:>14}{marker}"
        )
    return "\n".join(lines)
