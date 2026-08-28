"""Splitting, where a mistake is invisible and expensive.

A broken splitter does not raise. It produces a better metric, which is the one
kind of bug nobody investigates. These tests assert the properties that make a
split honest: nothing shared between parts, nothing lost, and — for the
temporal split — no training row from after a test row.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.splits import (
    DEFAULT_FRACTIONS,
    RANDOM_SEED,
    Split,
    group_split,
    random_split,
    temporal_split,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Twelve players across six seasons, two rows per season."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "season": [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
        }
    )


SEASONS = {"train_end_season": 2021, "validation_season": 2022, "test_start_season": 2023}


class TestTemporalSplit:
    def test_each_part_holds_the_seasons_it_should(self, frame: pd.DataFrame) -> None:
        split = temporal_split(frame, **SEASONS)
        assert set(frame.loc[split.train, "season"]) == {2019, 2020, 2021}
        assert set(frame.loc[split.validation, "season"]) == {2022}
        assert set(frame.loc[split.test, "season"]) == {2023, 2024}

    def test_no_training_row_comes_from_after_a_test_row(self, frame: pd.DataFrame) -> None:
        # The property the whole split exists for. If it fails, the model has
        # been shown the future and every reported metric is inflated.
        split = temporal_split(frame, **SEASONS)
        assert frame.loc[split.train, "season"].max() < frame.loc[split.test, "season"].min()

    def test_validation_sits_between_train_and_test(self, frame: pd.DataFrame) -> None:
        split = temporal_split(frame, **SEASONS)
        assert frame.loc[split.train, "season"].max() < frame.loc[split.validation, "season"].min()
        assert frame.loc[split.validation, "season"].max() < frame.loc[split.test, "season"].min()

    def test_a_player_may_appear_in_two_parts(self, frame: pd.DataFrame) -> None:
        # Expected, not a defect: a career spans the boundary. The leakage
        # report records it as a warning for exactly this reason.
        split = temporal_split(frame, **SEASONS)
        shared = set(frame.loc[split.train, "player_id"]) & set(frame.loc[split.test, "player_id"])
        assert shared

    def test_it_is_named_for_the_report(self, frame: pd.DataFrame) -> None:
        assert temporal_split(frame, **SEASONS).name == "temporal"


class TestGroupSplit:
    def test_no_player_appears_in_two_parts(self, frame: pd.DataFrame) -> None:
        split = group_split(frame)
        parts = [set(frame.loc[index, "player_id"]) for index in split.as_dict().values()]
        assert parts[0] & parts[1] == set()
        assert parts[0] & parts[2] == set()
        assert parts[1] & parts[2] == set()

    def test_it_is_reproducible(self, frame: pd.DataFrame) -> None:
        assert list(group_split(frame, seed=7).train) == list(group_split(frame, seed=7).train)

    def test_a_different_seed_gives_a_different_split(self) -> None:
        frame = pd.DataFrame({"player_id": range(100), "season": 2020})
        assert list(group_split(frame, seed=1).train) != list(group_split(frame, seed=2).train)


class TestRandomSplit:
    def test_it_is_reproducible(self, frame: pd.DataFrame) -> None:
        assert list(random_split(frame, seed=7).train) == list(random_split(frame, seed=7).train)

    def test_the_default_seed_is_the_project_seed(self, frame: pd.DataFrame) -> None:
        # The sibling project shipped 14 days of unreproducible metrics.
        assert list(random_split(frame).train) == list(random_split(frame, seed=RANDOM_SEED).train)

    def test_it_roughly_honours_the_requested_fractions(self) -> None:
        frame = pd.DataFrame({"player_id": range(1000), "season": 2020})
        split = random_split(frame)
        assert len(split.train) == pytest.approx(1000 * DEFAULT_FRACTIONS[0], abs=1)
        assert len(split.validation) == pytest.approx(1000 * DEFAULT_FRACTIONS[1], abs=1)


@pytest.mark.parametrize("splitter", [group_split, random_split])
def test_every_splitter_partitions_without_loss_or_overlap(
    splitter: object, frame: pd.DataFrame
) -> None:
    """No row in two parts, no row in none."""
    split = splitter(frame)  # type: ignore[operator]
    indices = list(split.as_dict().values())
    combined = indices[0].append(indices[1]).append(indices[2])

    assert len(combined) == len(frame)
    assert set(combined) == set(frame.index)
    assert not combined.duplicated().any()


def test_temporal_split_partitions_without_loss_when_seasons_are_contiguous(
    frame: pd.DataFrame,
) -> None:
    split = temporal_split(frame, **SEASONS)
    combined = split.train.append(split.validation).append(split.test)
    assert set(combined) == set(frame.index)
    assert not combined.duplicated().any()


def test_split_renders_its_sizes(frame: pd.DataFrame) -> None:
    split = temporal_split(frame, **SEASONS)
    assert split.sizes == {"train": 6, "validation": 2, "test": 4}
    assert "temporal" in split.render()


def test_as_dict_matches_what_the_leakage_validator_expects() -> None:
    split = Split("x", pd.Index([0]), pd.Index([1]), pd.Index([2]))
    assert set(split.as_dict()) == {"train", "validation", "test"}
