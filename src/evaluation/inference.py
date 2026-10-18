"""Uncertainty for metrics whose rows are not independent.

A player-season panel repeats players: pooled over two test seasons, about
half the players appear twice, and pooled over five backtest seasons some
appear five times. A paired *t*-test on absolute errors — how every p-value
this project published before Phase 20 was computed — treats each row as an
independent draw, so its standard error is too small by however much a
player's errors move together across seasons (audit F12).

The remedy here is the simplest one that respects that structure: resample
**players**, not rows, and recompute the mean over every row of each player
drawn. Seasons are not resampled — there are too few of them for a bootstrap
over seasons to mean anything — so a season-wide shock is part of what any
interval here describes, not something it averages away; results are reported
per season for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.splits import RANDOM_SEED

DRAWS = 2000
LEVEL = 0.95


@dataclass(frozen=True)
class Interval:
    """A mean, its player-clustered percentile interval, and what it rests on."""

    estimate: float
    lower: float
    upper: float
    rows: int
    clusters: int
    naive_standard_error: float
    """``std / sqrt(rows)``: what a row-level test would have used."""
    cluster_standard_error: float

    @property
    def design_effect(self) -> float:
        """Variance under clustering over variance treating rows as independent."""
        if self.naive_standard_error == 0:
            return float("nan")
        return float((self.cluster_standard_error / self.naive_standard_error) ** 2)

    @property
    def excludes_zero(self) -> bool:
        return bool(self.lower > 0 or self.upper < 0)

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "rows": self.rows,
            "clusters": self.clusters,
            "naive_standard_error": self.naive_standard_error,
            "cluster_standard_error": self.cluster_standard_error,
            "design_effect": self.design_effect,
            "excludes_zero": self.excludes_zero,
        }


def cluster_mean_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int = DRAWS,
    level: float = LEVEL,
    seed: int = RANDOM_SEED,
) -> Interval:
    """Mean of ``values`` with a bootstrap interval that resamples ``clusters``.

    Each draw takes as many clusters as there are, with replacement, and the
    mean over all of their rows — so a player drawn twice counts twice, and a
    player with two rows weighs twice a player with one, as in the point
    estimate.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if len(values) != len(clusters) or len(values) == 0:
        raise ValueError("values and clusters must be non-empty and the same length")

    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    k = len(sums)

    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, k, size=k)
        means[draw] = sums[chosen].sum() / counts[chosen].sum()

    tail = (1.0 - level) / 2.0
    return Interval(
        estimate=float(values.mean()),
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
        rows=len(values),
        clusters=k,
        naive_standard_error=(
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        ),
        cluster_standard_error=float(means.std(ddof=1)),
    )


def paired_difference(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    clusters: np.ndarray,
    **kwargs: float | int,
) -> Interval:
    """``mean(|a|) - mean(|b|)`` on the same rows, clustered by player.

    Positive means *a* errs more. Only for two predictions of the **same**
    rows; models scored on different populations are compared unpaired, each
    with its own interval, and never by differencing those.
    """
    a = np.abs(np.asarray(errors_a, dtype=float))
    b = np.abs(np.asarray(errors_b, dtype=float))
    if a.shape != b.shape:
        raise ValueError("a paired comparison needs both models scored on the same rows")
    return cluster_mean_interval(a - b, clusters, **kwargs)  # type: ignore[arg-type]


__all__ = ["DRAWS", "LEVEL", "Interval", "cluster_mean_interval", "paired_difference"]
