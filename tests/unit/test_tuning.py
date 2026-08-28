"""Hyperparameter search, and the fold structure it depends on.

The failure this file guards is a fold that trains on the future. It does not
raise; it returns a better score, and a better score is the one kind of bug
nobody investigates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.build import TARGET_COLUMN
from src.models.registry import MODEL_REGISTRY, ModelSpec
from src.models.tuning import MIN_TRAIN_SEASONS, season_folds, tune

NUMERIC = ["age", "goals"]
CATEGORICAL = ["position"]
FEATURES = ("age", "goals", "position")


@pytest.fixture
def seasons_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2012, 2022):
        for _ in range(40):
            goals = rng.integers(0, 20)
            rows.append(
                {
                    "season": season,
                    "age": rng.uniform(18, 34),
                    "goals": goals,
                    "position": rng.choice(["Attack", "Defender"]),
                    TARGET_COLUMN: float(np.expm1(13 + 0.1 * goals + rng.normal(0, 0.2))),
                }
            )
    return pd.DataFrame(rows)


class TestSeasonFolds:
    def test_every_fold_trains_only_on_earlier_seasons(self, seasons_frame: pd.DataFrame) -> None:
        """The property that makes this cross-validation and not leakage."""
        for fold in season_folds(seasons_frame, seasons_frame.index):
            train_max = seasons_frame.loc[fold.train, "season"].max()
            held_out = seasons_frame.loc[fold.validation, "season"].unique()

            assert list(held_out) == [fold.season]
            assert train_max < fold.season

    def test_the_window_expands(self, seasons_frame: pd.DataFrame) -> None:
        folds = season_folds(seasons_frame, seasons_frame.index)
        sizes = [len(fold.train) for fold in folds]
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)

    def test_the_first_fold_holds_back_enough_history(self, seasons_frame: pd.DataFrame) -> None:
        folds = season_folds(seasons_frame, seasons_frame.index)
        first_train_seasons = seasons_frame.loc[folds[0].train, "season"].nunique()
        assert first_train_seasons == MIN_TRAIN_SEASONS

    def test_one_fold_per_season_after_the_warmup(self, seasons_frame: pd.DataFrame) -> None:
        folds = season_folds(seasons_frame, seasons_frame.index)
        assert len(folds) == seasons_frame["season"].nunique() - MIN_TRAIN_SEASONS

    def test_too_few_seasons_yields_no_folds_rather_than_raising(self) -> None:
        # Small fixtures hit this legitimately; it means "no search possible",
        # not "something went wrong".
        frame = pd.DataFrame({"season": [2019, 2020]})
        assert season_folds(frame, frame.index) == []

    def test_folds_respect_the_supplied_training_index(self, seasons_frame: pd.DataFrame) -> None:
        """Folds must never reach outside the training rows they were given."""
        train_index = seasons_frame.index[seasons_frame["season"] <= 2018]
        for fold in season_folds(seasons_frame, train_index):
            assert set(fold.train) <= set(train_index)
            assert set(fold.validation) <= set(train_index)


class TestTune:
    def _tune(self, spec: ModelSpec, frame: pd.DataFrame) -> object:
        return tune(
            spec,
            frame,
            season_folds(frame, frame.index),
            feature_columns=FEATURES,
            numeric_features=NUMERIC,
            categorical_features=CATEGORICAL,
            target_column=TARGET_COLUMN,
        )

    def test_it_picks_a_configuration_from_the_grid(self, seasons_frame: pd.DataFrame) -> None:
        spec = MODEL_REGISTRY["ridge"]
        result = self._tune(spec, seasons_frame)

        assert result.model_name == "ridge"
        assert result.n_candidates == 3
        key, values = next(iter(spec.prefixed_grid().items()))
        assert result.best_params[key] in values

    def test_the_reported_score_is_a_real_mae_in_euros(self, seasons_frame: pd.DataFrame) -> None:
        result = self._tune(MODEL_REGISTRY["ridge"], seasons_frame)
        assert result.cv_mae > 1_000

    def test_a_family_without_a_grid_reports_one_candidate(
        self, seasons_frame: pd.DataFrame
    ) -> None:
        result = self._tune(MODEL_REGISTRY["linear"], seasons_frame)
        assert result.n_candidates == 1
        assert result.best_params == {}

    def test_it_is_reproducible(self, seasons_frame: pd.DataFrame) -> None:
        first = self._tune(MODEL_REGISTRY["ridge"], seasons_frame)
        second = self._tune(MODEL_REGISTRY["ridge"], seasons_frame)
        assert first == second

    def test_no_folds_returns_defaults_instead_of_raising(self) -> None:
        # A search that cannot run is a search with one candidate; the caller
        # still needs a fitted model.
        frame = pd.DataFrame(
            {
                "season": [2020],
                "age": [25.0],
                "goals": [1],
                "position": ["Attack"],
                TARGET_COLUMN: [1e6],
            }
        )
        result = tune(
            MODEL_REGISTRY["ridge"],
            frame,
            [],
            feature_columns=FEATURES,
            numeric_features=NUMERIC,
            categorical_features=CATEGORICAL,
            target_column=TARGET_COLUMN,
        )
        assert result.cv_mae != result.cv_mae  # NaN: no score was measured
        assert result.n_candidates == 3
