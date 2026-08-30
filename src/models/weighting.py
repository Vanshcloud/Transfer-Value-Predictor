"""How much each training row counts.

The audit's seventh limitation was season imbalance: 310 rows for 2011 against
5,674 for 2019. Widening the label window fixed most of it on its own — the
spread is now 1,093 to 7,760, a 7.1x ratio rather than 18.3x — but the question
of *which* correction helps was worth answering with a measurement rather than
an assumption.

Five schemes were fitted on the training seasons and scored on the validation
season, with the test seasons untouched until one had been chosen:

    none                validation MAE EUR 2,169,834
    inverse-frequency   validation MAE EUR 2,237,664   worse
    sqrt inverse-freq   validation MAE EUR 2,164,614
    recency             validation MAE EUR 2,126,386   best
    inverse x recency   validation MAE EUR 2,147,905

Inverse-frequency balancing — the obvious answer to "some seasons are small" —
was the *worst* of the five. That is the useful result: the imbalance is not
what hurts. Football's market inflates, squads turn over, and a 2011 row is a
weaker guide to 2024 than a 2020 row is, regardless of how many of each there
are. Up-weighting scarce old seasons makes the model more like the past, which
is the opposite of what a temporal split rewards.

Confirmed once on the test seasons: MAE EUR 2,364,999 -> 2,331,095, 1.43%
better. Small, and free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

RECENCY_BASE = 1.15
"""Weight multiplier per season of recency.

Chosen on the validation season, not tuned hard: 1.15 puts the most recent
training season roughly 4x the weight of the oldest across a ten-season span,
which is enough to express "newer is more relevant" without discarding the
early years. Larger bases were not measurably better and throw away data.
"""

SCHEMES = ("none", "inverse", "sqrt_inverse", "recency", "inverse_x_recency")


def season_weights(
    seasons: pd.Series, *, scheme: str = "recency", base: float = RECENCY_BASE
) -> np.ndarray | None:
    """Per-row training weights from the season each row belongs to.

    Returns ``None`` for ``"none"`` so a caller can pass the result straight
    through to ``fit`` without branching on the scheme.

    Every scheme is normalised to mean 1, so switching schemes changes the
    *relative* emphasis without also changing the effective learning rate —
    otherwise a comparison measures two things at once.

    Raises:
        ValueError: on an unknown scheme, rather than silently not weighting.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown weighting scheme {scheme!r}; expected one of {SCHEMES}")
    if scheme == "none":
        return None

    values = pd.to_numeric(seasons, errors="coerce")
    counts = values.value_counts()
    inverse = values.map(len(values) / (len(counts) * counts)).to_numpy(dtype=float)
    recency = np.power(base, values - values.min()).to_numpy(dtype=float)

    weights = {
        "inverse": inverse,
        "sqrt_inverse": np.sqrt(inverse),
        "recency": recency,
        "inverse_x_recency": inverse * recency,
    }[scheme]

    normalised: np.ndarray = weights / weights.mean()
    logger.info(
        "season weights (%s): %d rows, range %.2f-%.2f",
        scheme,
        len(normalised),
        float(normalised.min()),
        float(normalised.max()),
    )
    return normalised


def fit_params(seasons: pd.Series, *, scheme: str = "recency") -> dict[str, np.ndarray]:
    """Keyword arguments for ``Pipeline.fit``, empty when unweighted.

    The key is ``model__sample_weight`` rather than
    ``model__regressor__sample_weight``: the estimator step is a
    ``TransformedTargetRegressor``, which routes ``sample_weight`` to the
    wrapped regressor itself. Passing the longer path raises a TypeError from
    inside the booster, which is a confusing place to learn about routing.
    """
    weights = season_weights(seasons, scheme=scheme)
    return {} if weights is None else {"model__sample_weight": weights}


__all__ = ["RECENCY_BASE", "SCHEMES", "fit_params", "season_weights"]
