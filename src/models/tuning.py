"""Hyperparameter search that respects the arrow of time.

The search runs inside the training seasons only, on **expanding-window**
folds: fit on seasons up to N, score on season N+1, widen, repeat. K-fold would
train on 2023 to predict 2019 and return a number that cannot happen in
deployment — the same mistake as a random split, wearing cross-validation as a
disguise.

Selection is by MAE in EUR, not RMSE and not R^2. The question is "how far off
is a typical valuation"; RMSE would let a handful of EUR 100M outliers choose
the model. See docs/EXPERIMENT_TRACKING.md section 4.

Every fold is fitted with the same recency weighting the final model gets. That
sounds obvious and was not true until the final engineering pass: the search
fitted unweighted and only the winner was refitted weighted, which meant the
grid was ranked under an objective the project does not deploy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from src.evaluation.metrics import evaluate
from src.models.registry import ModelSpec, build_pipeline
from src.models.weighting import fit_params
from src.utils.logging import get_logger

logger = get_logger(__name__)

MIN_TRAIN_SEASONS = 3
"""Seasons in the first fold's training half. Fewer than three and the earliest
fold is fitting on a couple of thousand rows and scoring noise."""


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold: everything before a season, then that season."""

    train: pd.Index
    validation: pd.Index
    season: int


@dataclass(frozen=True)
class TuningResult:
    """The outcome of searching one model family."""

    model_name: str
    best_params: dict[str, Any]
    cv_mae: float
    n_candidates: int

    def render(self) -> str:
        return (
            f"{self.model_name:<18} cv MAE EUR {self.cv_mae:>12,.0f}  "
            f"({self.n_candidates} candidate(s))  {self.best_params or 'defaults'}"
        )


def season_folds(
    frame: pd.DataFrame,
    train_index: pd.Index,
    *,
    season_column: str = "season",
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> list[Fold]:
    """Expanding-window folds across the training seasons.

    Returns an empty list when there are too few seasons to hold one out, which
    the caller treats as "no search possible" rather than as an error — small
    fixtures and tests hit this legitimately.
    """
    rows = frame.loc[train_index]
    seasons = sorted(rows[season_column].unique())
    if len(seasons) <= min_train_seasons:
        return []

    folds = []
    for cut in range(min_train_seasons, len(seasons)):
        held_out = seasons[cut]
        folds.append(
            Fold(
                train=rows.index[rows[season_column] < held_out],
                validation=rows.index[rows[season_column] == held_out],
                season=int(held_out),
            )
        )
    return folds


def _candidates(spec: ModelSpec) -> list[dict[str, Any]]:
    """Every combination in the family's grid; a single empty one if it has none."""
    grid = spec.prefixed_grid()
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*grid.values())]


def _score_candidate(
    spec: ModelSpec,
    params: dict[str, Any],
    frame: pd.DataFrame,
    folds: list[Fold],
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    target_column: str,
) -> float:
    """Mean MAE across the folds for one hyperparameter combination.

    Fitted with the **same recency weights the final model is fitted with**.
    Until the final engineering pass it was not: the search fitted unweighted
    and the winner was then refitted weighted, so the grid was scored under an
    objective nobody deploys. A hyperparameter chosen for an unweighted fit is
    not necessarily the one an weighted fit wants — `min_child_samples` in
    particular trades directly against how sharply the recent seasons are
    up-weighted — and the difference cost nothing to remove.
    """
    features = list(feature_columns)
    errors = []

    for fold in folds:
        pipeline = build_pipeline(spec, numeric_features, categorical_features)
        pipeline.set_params(**params)

        train, validation = frame.loc[fold.train], frame.loc[fold.validation]
        pipeline.fit(train[features], train[target_column], **fit_params(train["season"]))
        predictions = pipeline.predict(validation[features])
        errors.append(evaluate(validation[target_column], predictions).mae)

    return float(sum(errors) / len(errors))


def tune(
    spec: ModelSpec,
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    target_column: str,
) -> TuningResult:
    """Search one family's grid and return its best configuration.

    With no folds, returns the family's defaults untuned rather than raising:
    a search that cannot run is not a failure, it is a search with one
    candidate, and the caller still needs a fitted model.
    """
    candidates = _candidates(spec)
    if not folds:
        return TuningResult(spec.name, candidates[0], float("nan"), len(candidates))

    scored = [
        (
            _score_candidate(
                spec,
                params,
                frame,
                folds,
                feature_columns,
                numeric_features,
                categorical_features,
                target_column,
            ),
            index,
            params,
        )
        for index, params in enumerate(candidates)
    ]
    # Tie-break on grid position, so a tie resolves the same way every run.
    best_mae, _, best_params = min(scored, key=lambda item: (item[0], item[1]))

    result = TuningResult(spec.name, best_params, best_mae, len(candidates))
    logger.info("tuned %s", result.render())
    return result
