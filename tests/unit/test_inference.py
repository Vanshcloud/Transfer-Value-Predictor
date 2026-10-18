"""Player-clustered uncertainty (audit F12)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.inference import cluster_mean_interval, paired_difference


def panel(players: int, seasons: int, player_sd: float, noise_sd: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, player_sd, players)
    clusters = np.repeat(np.arange(players), seasons)
    values = effect[clusters] + rng.normal(0, noise_sd, players * seasons)
    return values, clusters


class TestClusterInterval:
    def test_independent_rows_need_no_correction(self) -> None:
        values, clusters = panel(2000, 1, player_sd=0.0, noise_sd=1.0)
        interval = cluster_mean_interval(values, clusters, draws=500)
        assert interval.design_effect == pytest.approx(1.0, abs=0.25)
        assert interval.lower < interval.estimate < interval.upper

    def test_repeated_players_widen_the_interval(self) -> None:
        """Five seasons of the same players, errors that move together: a
        row-level standard error is far too small, and the design effect says
        by how much."""
        values, clusters = panel(400, 5, player_sd=1.0, noise_sd=0.5)
        interval = cluster_mean_interval(values, clusters, draws=500)
        assert interval.clusters == 400 and interval.rows == 2000
        assert interval.design_effect > 3.0
        assert interval.cluster_standard_error > interval.naive_standard_error

    def test_it_is_reproducible(self) -> None:
        values, clusters = panel(300, 2, 1.0, 1.0)
        assert cluster_mean_interval(values, clusters) == cluster_mean_interval(values, clusters)

    def test_mismatched_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError):
            cluster_mean_interval(np.ones(3), np.ones(2))


class TestPairedDifference:
    def test_a_real_difference_is_detected_and_signed(self) -> None:
        rng = np.random.default_rng(1)
        clusters = np.repeat(np.arange(500), 2)
        b = rng.normal(0, 1, 1000)
        a = b * 1.5  # a errs more on every row
        difference = paired_difference(a, b, clusters, draws=500)
        assert difference.estimate > 0 and difference.excludes_zero

    def test_no_difference_is_not_called_one(self) -> None:
        rng = np.random.default_rng(2)
        clusters = np.repeat(np.arange(500), 2)
        a = rng.normal(0, 1, 1000)
        b = rng.permutation(a)
        difference = paired_difference(a, b, clusters, draws=500)
        assert not difference.excludes_zero

    def test_different_populations_cannot_be_paired(self) -> None:
        with pytest.raises(ValueError, match="same rows"):
            paired_difference(np.ones(4), np.ones(3), np.arange(4))
