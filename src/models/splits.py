"""Three ways to divide the table, and only one of them is the headline.

The spike measured all three on this data and the result corrected a prior
assumption. The expected danger — the same player appearing in both train and
test — turned out to be worth almost nothing (R^2 0.805 random vs 0.795
grouped), because the as-of join already yields one row per player-season. The
danger that *does* matter is temporal: R^2 0.770 and a EUR MAE 45% worse than
the grouped split's.

Those three figures come from ``scripts/train_baseline.py`` on the current
table and are restated whenever it changes. They were 0.465 / 0.455 / 0.412 and
"roughly 60% worse" when this docstring was first written, against a table a
third the size; the ordering has never changed, which is the part the argument
rests on.

So the temporal split is the reported number and the other two exist to be
compared against it. A random split on a time series answers "how well does
this model interpolate among seasons it has already seen", which is not a
question anyone deploying it will ever ask.

Splitting three ways rather than two is what keeps that honest: the validation
season tunes, and the test seasons are touched once, at the end. A test set
consulted repeatedly is a validation set wearing a disguise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

RANDOM_SEED = 42
"""Every stochastic step in this project derives from this one number.

The sibling project shipped fourteen days of unreproducible metrics before
someone noticed a missing seed. Two runs here must agree exactly.
"""

DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
"""Train/validation/test proportions for the two non-temporal splitters.

Chosen to sit near the temporal split's natural shape so the three are roughly
comparable. They are diagnostics, not the headline, so the exact figures matter
less than holding them fixed between runs.
"""


@dataclass(frozen=True)
class Split:
    """Row labels for one three-way division of a table."""

    name: str
    train: pd.Index
    validation: pd.Index
    test: pd.Index

    def as_dict(self) -> dict[str, pd.Index]:
        """The form :class:`~src.validation.leakage.LeakageValidator` expects."""
        return {"train": self.train, "validation": self.validation, "test": self.test}

    @property
    def sizes(self) -> dict[str, int]:
        return {name: len(index) for name, index in self.as_dict().items()}

    def render(self) -> str:
        counts = "  ".join(f"{name} {size:,}" for name, size in self.sizes.items())
        return f"{self.name}: {counts}"


Splitter = Callable[[pd.DataFrame], Split]
"""What every splitter is. A type alias, not a Protocol — there is one method."""


def temporal_split(
    frame: pd.DataFrame,
    *,
    train_end_season: int,
    validation_season: int,
    test_start_season: int,
    season_column: str = "season",
) -> Split:
    """Split by season. The only split whose metrics are reported as headline.

    Train on everything up to ``train_end_season``, tune on
    ``validation_season``, and report on ``test_start_season`` onward. This is
    the only division that matches deployment, where the model has never seen
    the season it is asked about.
    """
    season = frame[season_column]
    return Split(
        name="temporal",
        train=frame.index[season <= train_end_season],
        validation=frame.index[season == validation_season],
        test=frame.index[season >= test_start_season],
    )


def _partition(items: np.ndarray, fractions: tuple[float, float, float]) -> list[np.ndarray]:
    train_end = int(len(items) * fractions[0])
    validation_end = train_end + int(len(items) * fractions[1])
    return [items[:train_end], items[train_end:validation_end], items[validation_end:]]


def group_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "player_id",
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = RANDOM_SEED,
) -> Split:
    """Split by player, so no player appears in two parts.

    A diagnostic. Comparing it against the random split measures how much the
    model gains from having seen a player before; the spike measured that gain
    as almost nothing, which is why the temporal comparison is the one that
    matters.
    """
    groups = frame[group_column].unique()
    shuffled = np.random.default_rng(seed).permutation(groups)
    train_groups, validation_groups, test_groups = _partition(shuffled, fractions)

    return Split(
        name="group",
        train=frame.index[frame[group_column].isin(train_groups)],
        validation=frame.index[frame[group_column].isin(validation_groups)],
        test=frame.index[frame[group_column].isin(test_groups)],
    )


def random_split(
    frame: pd.DataFrame,
    *,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = RANDOM_SEED,
) -> Split:
    """Split rows at random, ignoring both player and season.

    The most flattering split and the least informative one. Kept only so the
    gap to the temporal number can be quoted rather than asserted.
    """
    shuffled = np.random.default_rng(seed).permutation(frame.index.to_numpy())
    train, validation, test = _partition(shuffled, fractions)

    return Split(
        name="random",
        train=pd.Index(train),
        validation=pd.Index(validation),
        test=pd.Index(test),
    )
