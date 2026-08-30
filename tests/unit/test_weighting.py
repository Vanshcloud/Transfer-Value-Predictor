"""Season weighting: the schemes, and the property each must hold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.weighting import SCHEMES, fit_params, season_weights


@pytest.fixture
def seasons() -> pd.Series:
    """Deliberately imbalanced: 2011 is scarce, 2020 is abundant."""
    return pd.Series([2011] * 10 + [2015] * 50 + [2020] * 200)


class TestSeasonWeights:
    def test_none_returns_none_so_callers_need_no_branch(self, seasons: pd.Series) -> None:
        assert season_weights(seasons, scheme="none") is None

    @pytest.mark.parametrize("scheme", [s for s in SCHEMES if s != "none"])
    def test_every_scheme_normalises_to_mean_one(self, seasons: pd.Series, scheme: str) -> None:
        """Otherwise switching schemes also changes the effective learning
        rate, and a comparison between them measures two things at once."""
        weights = season_weights(seasons, scheme=scheme)
        assert weights is not None
        assert float(np.mean(weights)) == pytest.approx(1.0)

    def test_inverse_frequency_equalises_the_seasons(self, seasons: pd.Series) -> None:
        weights = season_weights(seasons, scheme="inverse")
        assert weights is not None
        totals = pd.DataFrame({"season": seasons, "w": weights}).groupby("season")["w"].sum()
        assert totals.max() == pytest.approx(totals.min())

    def test_recency_favours_later_seasons(self, seasons: pd.Series) -> None:
        weights = season_weights(seasons, scheme="recency")
        assert weights is not None
        per = pd.DataFrame({"season": seasons, "w": weights}).groupby("season")["w"].first()
        assert per[2020] > per[2015] > per[2011]

    def test_recency_ignores_how_many_rows_a_season_has(self) -> None:
        """It is a statement about age, not about abundance."""
        a = season_weights(pd.Series([2011] * 5 + [2020] * 5), scheme="recency")
        b = season_weights(pd.Series([2011] * 5 + [2020] * 500), scheme="recency")
        assert a is not None and b is not None
        assert a.max() / a.min() == pytest.approx(b.max() / b.min())

    def test_an_unknown_scheme_raises_rather_than_silently_not_weighting(self) -> None:
        with pytest.raises(ValueError, match="unknown weighting scheme"):
            season_weights(pd.Series([2020]), scheme="magic")

    def test_a_single_season_weights_everything_equally(self) -> None:
        weights = season_weights(pd.Series([2020] * 7), scheme="recency")
        assert weights is not None
        assert np.allclose(weights, 1.0)


class TestFitParams:
    def test_the_key_matches_what_the_pipeline_routes(self) -> None:
        """`model__regressor__sample_weight` raises a TypeError from inside the
        booster: the estimator step is a TransformedTargetRegressor, which
        routes sample_weight to the wrapped regressor itself."""
        params = fit_params(pd.Series([2019, 2020]), scheme="recency")
        assert set(params) == {"model__sample_weight"}

    def test_unweighted_produces_no_keyword_at_all(self) -> None:
        assert fit_params(pd.Series([2019, 2020]), scheme="none") == {}
