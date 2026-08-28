"""Feature engineering — the join the whole project rests on.

Two kinds of test here. Small synthetic frames pin down each transformation's
exact behaviour, because a season boundary or a merge direction is easy to get
subtly wrong and impossible to eyeball on 37,000 rows. The sample-data tests
then assert the invariants that must survive contact with real, messy input —
above all that no row was built from information that postdates its label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.build import (
    AS_OF_COLUMN,
    CAREER_CENSORING_CEILING,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_TIME_COLUMN,
    LABEL_TIME_COLUMN,
    MINUTES_FLOOR,
    NUMERIC_FEATURES,
    PRIOR_VALUE_FEATURES,
    TARGET_COLUMN,
    add_career_history,
    add_derived_features,
    aggregate_appearances,
    assign_season,
    attach_label,
    attach_player_attributes,
    build_training_table,
    season_boundary_dates,
    select_variant,
)
from src.validation.leakage import CURRENT_STATE_COLUMNS, detect_leakage


def make_appearances(rows: list[dict[str, object]]) -> pd.DataFrame:
    """An appearances frame with the columns the aggregator needs."""
    defaults = {
        "appearance_id": "a",
        "player_id": 1,
        "date": "2021-09-01",
        "goals": 0,
        "assists": 0,
        "minutes_played": 90,
        "yellow_cards": 0,
        "red_cards": 0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows]).assign(
        appearance_id=lambda f: [f"a{i}" for i in range(len(f))]
    )


# --------------------------------------------------------------------------
# Season assignment
# --------------------------------------------------------------------------


class TestAssignSeason:
    def test_august_starts_the_season_named_for_that_year(self) -> None:
        dates = pd.Series(pd.to_datetime(["2021-08-01", "2021-12-31"]))
        assert list(assign_season(dates)) == [2021, 2021]

    def test_january_to_july_belong_to_the_previous_year(self) -> None:
        dates = pd.Series(pd.to_datetime(["2022-01-15", "2022-07-31"]))
        assert list(assign_season(dates)) == [2021, 2021]

    def test_the_boundary_is_exactly_the_first_of_august(self) -> None:
        dates = pd.Series(pd.to_datetime(["2022-07-31", "2022-08-01"]))
        assert list(assign_season(dates)) == [2021, 2022]

    @pytest.mark.filterwarnings("ignore:Could not infer format")
    def test_an_unparseable_date_gets_no_season_rather_than_a_wrong_one(self) -> None:
        seasons = assign_season(pd.Series(["not-a-date", "2021-09-01"]))
        assert pd.isna(seasons.iloc[0])
        assert seasons.iloc[1] == 2021

    def test_start_month_is_configurable(self) -> None:
        dates = pd.Series(pd.to_datetime(["2021-06-15"]))
        assert assign_season(dates, start_month=1).iloc[0] == 2021
        assert assign_season(dates, start_month=8).iloc[0] == 2020


class TestSeasonBoundary:
    def test_an_august_season_ends_on_the_first_of_july_following(self) -> None:
        boundary = season_boundary_dates(pd.Series([2021, 2015]))
        assert list(boundary) == [pd.Timestamp("2022-07-01"), pd.Timestamp("2016-07-01")]

    def test_a_january_season_ends_in_december_of_the_same_year(self) -> None:
        # Guards the month-zero wraparound rather than the calendar: config
        # allows season_start_month=1 and it must not produce month 0.
        boundary = season_boundary_dates(pd.Series([2021]), start_month=1)
        assert boundary.iloc[0] == pd.Timestamp("2021-12-01")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


class TestAggregateAppearances:
    def test_one_row_per_player_season_with_summed_totals(self) -> None:
        frame = make_appearances(
            [
                {"date": "2021-09-01", "goals": 2, "assists": 1, "minutes_played": 90},
                {"date": "2021-10-01", "goals": 1, "assists": 0, "minutes_played": 45},
                {"date": "2022-09-01", "goals": 5, "assists": 3, "minutes_played": 90},
            ]
        )
        result = aggregate_appearances(frame).sort_values("season")

        assert len(result) == 2
        assert list(result["season"]) == [2021, 2022]
        assert list(result["appearances"]) == [2, 1]
        assert list(result["goals"]) == [3, 5]
        assert list(result["minutes_played"]) == [135, 90]

    def test_players_are_kept_separate(self) -> None:
        frame = make_appearances(
            [
                {"player_id": 1, "date": "2021-09-01", "goals": 3},
                {"player_id": 2, "date": "2021-09-01", "goals": 7},
            ]
        )
        result = aggregate_appearances(frame).sort_values("player_id")
        assert list(result["goals"]) == [3, 7]

    def test_last_appearance_date_is_the_latest_match(self) -> None:
        frame = make_appearances(
            [{"date": "2021-09-01"}, {"date": "2022-05-20"}, {"date": "2021-11-03"}]
        )
        result = aggregate_appearances(frame)
        assert result[FEATURE_TIME_COLUMN].iloc[0] == pd.Timestamp("2022-05-20")

    def test_as_of_date_is_the_season_boundary_when_the_season_ends_early(self) -> None:
        frame = make_appearances([{"date": "2022-05-20"}])
        result = aggregate_appearances(frame)
        assert result[AS_OF_COLUMN].iloc[0] == pd.Timestamp("2022-07-01")

    def test_as_of_date_follows_a_season_still_being_played_in_july(self) -> None:
        # Summer-calendar leagues and continental tournaments run past 1 July.
        # Anchoring them on the boundary would let a valuation published
        # mid-tournament label a row whose features run past it.
        frame = make_appearances([{"date": "2022-07-20"}])
        result = aggregate_appearances(frame)
        assert result[AS_OF_COLUMN].iloc[0] == pd.Timestamp("2022-07-20")

    def test_years_since_debut_measures_from_the_first_ever_appearance(self) -> None:
        frame = make_appearances(
            [{"date": "2018-09-01"}, {"date": "2021-09-01"}, {"date": "2021-11-01"}]
        )
        result = aggregate_appearances(frame).sort_values("season")
        # Season 2018 anchors on 2019-07-01, season 2021 on 2022-07-01; the
        # debut is 2018-09-01 in both cases, not the season's own first match.
        assert result["years_since_debut"].iloc[0] == pytest.approx(0.83, abs=0.02)
        assert result["years_since_debut"].iloc[1] == pytest.approx(3.83, abs=0.02)

    def test_years_since_debut_is_capped_at_the_censoring_ceiling(self) -> None:
        # Uncapped, this feature's maximum runs 0, 1, 2 ... in lockstep with
        # the season, because the dataset only starts watching in 2012 — which
        # rebuilds the calendar variable the project excluded on purpose.
        frame = make_appearances([{"date": "2000-09-01"}, {"date": "2021-09-01"}])
        result = aggregate_appearances(frame)
        assert result["years_since_debut"].max() == CAREER_CENSORING_CEILING

    def test_years_since_debut_is_never_negative(self) -> None:
        frame = make_appearances([{"date": "2021-09-01"}, {"date": "2022-07-20"}])
        result = aggregate_appearances(frame)
        assert (result["years_since_debut"] >= 0).all()

    def test_debut_is_not_shared_between_players(self) -> None:
        """Two players in the same season, six years apart in career stage."""
        frame = make_appearances(
            [
                {"player_id": 1, "date": "2015-09-01"},
                {"player_id": 1, "date": "2021-09-01"},
                {"player_id": 2, "date": "2021-09-01"},
            ]
        )
        result = aggregate_appearances(frame)
        in_2021 = result[result["season"] == 2021].set_index("player_id")
        assert in_2021.loc[1, "years_since_debut"] == pytest.approx(6.83, abs=0.02)
        assert in_2021.loc[2, "years_since_debut"] == pytest.approx(0.83, abs=0.02)

    def test_as_of_date_is_never_before_the_last_appearance(self) -> None:
        frame = make_appearances(
            [{"date": "2021-09-01"}, {"date": "2022-07-20"}, {"date": "2022-06-01"}]
        )
        result = aggregate_appearances(frame)
        assert (result[AS_OF_COLUMN] >= result[FEATURE_TIME_COLUMN]).all()


# --------------------------------------------------------------------------
# The label join
# --------------------------------------------------------------------------


@pytest.fixture
def one_player_season() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1],
            "season": [2021],
            "appearances": [10],
            FEATURE_TIME_COLUMN: [pd.Timestamp("2022-05-01")],
            AS_OF_COLUMN: [pd.Timestamp("2022-07-01")],
        }
    )


def valuations(dates: list[str], values: list[int], player_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame({"player_id": player_id, "date": dates, TARGET_COLUMN: values})


class TestAttachLabel:
    def test_the_label_is_the_first_valuation_after_the_as_of_date(
        self, one_player_season: pd.DataFrame
    ) -> None:
        result = attach_label(
            one_player_season,
            valuations(["2022-07-15", "2022-09-01"], [5_000_000, 9_000_000]),
        )
        assert result[TARGET_COLUMN].iloc[0] == 5_000_000
        assert result[LABEL_TIME_COLUMN].iloc[0] == pd.Timestamp("2022-07-15")

    def test_a_valuation_before_the_as_of_date_is_never_used(
        self, one_player_season: pd.DataFrame
    ) -> None:
        # The failure this guards is a backward merge: it would label the
        # season with a value set halfway through it, which the features
        # already describe.
        result = attach_label(one_player_season, valuations(["2022-01-01"], [5_000_000]))
        assert result.empty

    def test_a_valuation_beyond_the_tolerance_is_dropped_not_stretched(
        self, one_player_season: pd.DataFrame
    ) -> None:
        # 2022-07-01 + 120 days = 2022-10-29.
        assert attach_label(one_player_season, valuations(["2022-10-30"], [1])).empty
        assert len(attach_label(one_player_season, valuations(["2022-10-29"], [1]))) == 1

    def test_a_valuation_exactly_on_the_as_of_date_counts(
        self, one_player_season: pd.DataFrame
    ) -> None:
        result = attach_label(one_player_season, valuations(["2022-07-01"], [3_000_000]))
        assert result[TARGET_COLUMN].iloc[0] == 3_000_000

    def test_labels_do_not_cross_players(self, one_player_season: pd.DataFrame) -> None:
        result = attach_label(
            one_player_season, valuations(["2022-07-15"], [5_000_000], player_id=999)
        )
        assert result.empty

    def test_a_player_season_with_no_valuation_is_dropped(
        self, one_player_season: pd.DataFrame
    ) -> None:
        assert attach_label(one_player_season, valuations([], [])).empty


# --------------------------------------------------------------------------
# Player attributes
# --------------------------------------------------------------------------


@pytest.fixture
def labelled_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1],
            "season": [2021],
            AS_OF_COLUMN: [pd.Timestamp("2022-07-01")],
            LABEL_TIME_COLUMN: [pd.Timestamp("2022-08-01")],
            TARGET_COLUMN: [5_000_000],
        }
    )


def players_frame(**overrides: object) -> pd.DataFrame:
    row = {
        "player_id": 1,
        "date_of_birth": "2000-07-01",
        "position": "Attack",
        "sub_position": "Centre-Forward",
        "foot": "right",
        "height_in_cm": 180.0,
        "country_of_citizenship": "Brazil",
        # Current state — present in players.csv, must not survive the join.
        "contract_expiration_date": "2027-06-30",
        "market_value_in_eur": 80_000_000.0,
        "current_club_name": "Somewhere FC",
        "agent_name": "An Agent",
    }
    return pd.DataFrame([{**row, **overrides}])


class TestAttachPlayerAttributes:
    def test_age_is_measured_at_the_as_of_date(self, labelled_row: pd.DataFrame) -> None:
        result = attach_player_attributes(labelled_row, players_frame())
        assert result["age"].iloc[0] == pytest.approx(22.0, abs=0.01)

    def test_current_state_columns_do_not_survive_the_join(
        self, labelled_row: pd.DataFrame
    ) -> None:
        # The single most important assertion in this file. players.csv
        # describes the player *now*; every one of these columns on a 2015 row
        # states a fact from 2026.
        result = attach_player_attributes(labelled_row, players_frame())
        # market_value_in_eur is itself a member of CURRENT_STATE_COLUMNS, and
        # it is legitimately here as the *label*. Everything else must be gone.
        assert set(result.columns) & CURRENT_STATE_COLUMNS == {TARGET_COLUMN}

    def test_a_player_without_a_birth_date_is_dropped(self, labelled_row: pd.DataFrame) -> None:
        result = attach_player_attributes(labelled_row, players_frame(date_of_birth=None))
        assert result.empty

    def test_a_duplicated_player_id_raises_rather_than_multiplying_rows(
        self, labelled_row: pd.DataFrame
    ) -> None:
        duplicated = pd.concat([players_frame(), players_frame()], ignore_index=True)
        with pytest.raises(pd.errors.MergeError):
            attach_player_attributes(labelled_row, duplicated)


# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------


class TestDerivedFeatures:
    def base(self, **overrides: object) -> pd.DataFrame:
        row = {
            "goals": 10,
            "assists": 5,
            "yellow_cards": 3,
            "red_cards": 1,
            "minutes_played": 1800,
            "appearances": 20,
            "age": 25.0,
        }
        return pd.DataFrame([{**row, **overrides}])

    def test_per_90_rates_are_scaled_by_minutes(self) -> None:
        result = add_derived_features(self.base())
        assert result["goals_per_90"].iloc[0] == pytest.approx(0.5)
        assert result["assists_per_90"].iloc[0] == pytest.approx(0.25)
        assert result["cards_per_90"].iloc[0] == pytest.approx(0.2)

    def test_a_tiny_cameo_cannot_produce_an_absurd_rate(self) -> None:
        # One goal in three minutes is 30 goals per 90 unclipped, and that one
        # row then outweighs a whole season from a real striker.
        result = add_derived_features(self.base(goals=1, minutes_played=3, appearances=1))
        assert result["goals_per_90"].iloc[0] == pytest.approx(90.0 / MINUTES_FLOOR)

    def test_the_floor_does_not_touch_a_full_season(self) -> None:
        assert MINUTES_FLOOR == 90.0
        result = add_derived_features(self.base(minutes_played=900, goals=10))
        assert result["goals_per_90"].iloc[0] == pytest.approx(1.0)

    def test_minutes_per_appearance_never_divides_by_zero(self) -> None:
        result = add_derived_features(self.base(appearances=0, minutes_played=0))
        assert result["minutes_per_appearance"].iloc[0] == 0.0
        assert np.isfinite(result["minutes_per_appearance"]).all()

    def test_age_squared_bends_the_age_curve(self) -> None:
        result = add_derived_features(self.base(age=25.0))
        assert result["age_squared"].iloc[0] == pytest.approx(625.0)


# --------------------------------------------------------------------------
# The lagged prior value
# --------------------------------------------------------------------------


def career(seasons: list[int], values: list[int], label_dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": 1,
            "season": seasons,
            TARGET_COLUMN: values,
            LABEL_TIME_COLUMN: pd.to_datetime(label_dates),
        }
    )


class TestCareerHistory:
    def test_the_first_season_has_no_prior_value(self) -> None:
        result = add_career_history(career([2019], [1_000_000], ["2020-07-15"]))
        assert pd.isna(result["prev_market_value_in_eur"].iloc[0])

    def test_a_later_season_carries_the_previous_label(self) -> None:
        result = add_career_history(
            career([2019, 2020], [1_000_000, 4_000_000], ["2020-07-15", "2021-07-15"])
        ).sort_values("season")
        assert result["prev_market_value_in_eur"].iloc[1] == 1_000_000
        assert result["prev_value_age_days"].iloc[1] == 365

    def test_seasons_observed_counts_only_earlier_rows(self) -> None:
        result = add_career_history(
            career([2018, 2019, 2020], [1, 2, 3], ["2019-07-15", "2020-07-15", "2021-07-15"])
        ).sort_values("season")
        assert list(result["seasons_observed"]) == [0, 1, 2]

    def test_seasons_observed_is_capped_too(self) -> None:
        seasons = list(range(2000, 2020))
        result = add_career_history(
            career(seasons, list(range(len(seasons))), [f"{y + 1}-07-15" for y in seasons])
        )
        assert result["seasons_observed"].max() == int(CAREER_CENSORING_CEILING)

    def test_seasons_observed_restarts_for_each_player(self) -> None:
        frame = pd.DataFrame(
            {
                "player_id": [1, 1, 2],
                "season": [2019, 2020, 2020],
                TARGET_COLUMN: [1, 2, 3],
                LABEL_TIME_COLUMN: pd.to_datetime(["2020-07-15", "2021-07-15", "2021-07-15"]),
            }
        )
        result = add_career_history(frame).sort_values(["player_id", "season"])
        assert list(result["seasons_observed"]) == [0, 1, 0]

    def test_a_gap_year_is_carried_with_its_staleness_recorded(self) -> None:
        # Not dropped: a three-year-old valuation still informs, as long as the
        # model is told how old it is.
        result = add_career_history(
            career([2016, 2020], [1_000_000, 4_000_000], ["2017-07-15", "2021-07-15"])
        ).sort_values("season")
        assert result["prev_market_value_in_eur"].iloc[1] == 1_000_000
        assert result["prev_value_age_days"].iloc[1] == pytest.approx(1461, abs=2)

    def test_the_prior_label_always_predates_the_current_one(self) -> None:
        result = add_career_history(
            career([2018, 2019, 2020], [1, 2, 3], ["2019-07-15", "2020-07-15", "2021-07-15"])
        )
        known = result["prev_value_age_days"].dropna()
        assert (known > 0).all()

    def test_values_do_not_leak_between_players(self) -> None:
        frame = pd.DataFrame(
            {
                "player_id": [1, 2],
                "season": [2020, 2021],
                TARGET_COLUMN: [1_000_000, 9_000_000],
                LABEL_TIME_COLUMN: pd.to_datetime(["2021-07-15", "2022-07-15"]),
            }
        )
        result = add_career_history(frame)
        assert result["prev_market_value_in_eur"].isna().all()

    def test_input_order_does_not_change_the_result(self) -> None:
        rows = career([2018, 2019, 2020], [1, 2, 3], ["2019-07-15", "2020-07-15", "2021-07-15"])
        forward = add_career_history(rows).sort_values("season")
        shuffled = add_career_history(rows.iloc[::-1]).sort_values("season")
        assert list(forward["prev_market_value_in_eur"].fillna(-1)) == list(
            shuffled["prev_market_value_in_eur"].fillna(-1)
        )


# --------------------------------------------------------------------------
# Variant selection
# --------------------------------------------------------------------------


class TestSelectVariant:
    @pytest.fixture
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": [1, 1, 2],
                "prev_market_value_in_eur": [np.nan, 1_000_000.0, np.nan],
            }
        )

    def test_performance_only_keeps_every_row(self, table: pd.DataFrame) -> None:
        rows, columns = select_variant(table, include_prior_value=False)
        assert len(rows) == 3
        assert columns == FEATURE_COLUMNS

    def test_with_prior_value_keeps_only_rows_that_have_one(self, table: pd.DataFrame) -> None:
        rows, columns = select_variant(table, include_prior_value=True)
        assert len(rows) == 1
        assert set(PRIOR_VALUE_FEATURES) <= set(columns)

    def test_the_target_is_not_a_feature_in_either_variant(self) -> None:
        assert TARGET_COLUMN not in FEATURE_COLUMNS
        assert TARGET_COLUMN not in PRIOR_VALUE_FEATURES

    def test_no_feature_is_a_current_state_column(self) -> None:
        assert not set(FEATURE_COLUMNS + PRIOR_VALUE_FEATURES) & CURRENT_STATE_COLUMNS

    def test_season_is_not_a_feature(self) -> None:
        # It is the split key. A model that learns "later season -> higher
        # value" extrapolates straight off the end of its training range.
        assert "season" not in FEATURE_COLUMNS

    def test_career_stage_features_replace_what_raw_season_would_carry(self) -> None:
        # Dropping `season` should not mean dropping time. These two answer
        # "how far into a career is this" without encoding which calendar year
        # it is, so they stay meaningful on seasons the model never trained on.
        assert {"years_since_debut", "seasons_observed"} <= set(FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# The whole table, on real (sampled) data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_table(
    sample_players: pd.DataFrame,
    sample_valuations: pd.DataFrame,
    sample_appearances: pd.DataFrame,
) -> pd.DataFrame:
    return build_training_table(sample_players, sample_valuations, sample_appearances)


class TestBuildTrainingTable:
    def test_it_produces_rows_from_real_data(self, sample_table: pd.DataFrame) -> None:
        assert not sample_table.empty
        assert set(FEATURE_COLUMNS) <= set(sample_table.columns)
        assert TARGET_COLUMN in sample_table.columns

    def test_no_feature_was_observed_after_its_label(self, sample_table: pd.DataFrame) -> None:
        # The verification the whole phase exists to satisfy. If this fails,
        # every metric downstream is inflated by an unknown amount.
        assert (sample_table[FEATURE_TIME_COLUMN] <= sample_table[LABEL_TIME_COLUMN]).all()

    def test_one_row_per_player_season(self, sample_table: pd.DataFrame) -> None:
        assert not sample_table.duplicated(subset=["player_id", "season"]).any()

    def test_the_label_is_always_a_real_positive_amount(self, sample_table: pd.DataFrame) -> None:
        assert sample_table[TARGET_COLUMN].notna().all()
        assert (sample_table[TARGET_COLUMN] > 0).all()

    def test_core_features_are_never_null(self, sample_table: pd.DataFrame) -> None:
        for column in ("age", "position", "minutes_played", "appearances"):
            assert sample_table[column].notna().all(), column

    def test_numeric_features_are_numeric_and_finite(self, sample_table: pd.DataFrame) -> None:
        numeric = sample_table.loc[:, list(NUMERIC_FEATURES)]
        assert numeric.apply(pd.api.types.is_numeric_dtype).all()
        values = numeric.to_numpy(dtype=float)
        assert np.isfinite(values[~np.isnan(values)]).all()

    def test_categorical_features_survive_as_labels_not_codes(
        self, sample_table: pd.DataFrame
    ) -> None:
        assert (
            sample_table["position"]
            .isin({"Attack", "Defender", "Goalkeeper", "Midfield", "Missing"})
            .all()
        )
        assert set(CATEGORICAL_FEATURES) <= set(sample_table.columns)

    def test_the_leakage_stage_passes_on_both_variants(self, sample_table: pd.DataFrame) -> None:
        for include_prior in (False, True):
            rows, columns = select_variant(sample_table, include_prior_value=include_prior)
            report = detect_leakage(
                rows,
                feature_columns=columns,
                target_column=TARGET_COLUMN,
                feature_time_column=FEATURE_TIME_COLUMN,
                label_time_column=LABEL_TIME_COLUMN,
            )
            assert report.ok, report.render()

    def test_the_build_is_deterministic(
        self,
        sample_players: pd.DataFrame,
        sample_valuations: pd.DataFrame,
        sample_appearances: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> None:
        again = build_training_table(sample_players, sample_valuations, sample_appearances)
        pd.testing.assert_frame_equal(sample_table, again)

    def test_a_leaky_table_is_caught_rather_than_assumed_impossible(
        self, sample_table: pd.DataFrame
    ) -> None:
        # Proves the previous assertions are testing something: break the
        # ordering deliberately and the detector must fire.
        leaky = sample_table.copy()
        leaky.loc[leaky.index[0], FEATURE_TIME_COLUMN] = leaky[LABEL_TIME_COLUMN].iloc[
            0
        ] + pd.Timedelta(1, "D")
        report = detect_leakage(
            leaky,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            feature_time_column=FEATURE_TIME_COLUMN,
            label_time_column=LABEL_TIME_COLUMN,
        )
        assert not report.ok


# --------------------------------------------------------------------------
# The pipeline stage
# --------------------------------------------------------------------------


@pytest.fixture
def sample_store(tmp_path: object) -> object:
    from pathlib import Path

    from src.storage.duckdb_store import DuckDBParquetStore
    from src.utils.paths import PROJECT_ROOT

    store = DuckDBParquetStore(Path(str(tmp_path)) / "processed")
    for name in ("players", "player_valuations", "appearances"):
        store.write_table(name, pd.read_csv(PROJECT_ROOT / "data" / "sample" / f"{name}.csv"))
    return store


class TestFeaturePipeline:
    def test_the_stage_writes_the_training_table(self, sample_store: object) -> None:
        from src.pipelines.features import TRAINING_TABLE, build_features

        report = build_features(sample_store)  # type: ignore[arg-type]

        assert sample_store.has_table(TRAINING_TABLE)  # type: ignore[attr-defined]
        stored = sample_store.read_table(TRAINING_TABLE)  # type: ignore[attr-defined]
        assert len(stored) == report.rows
        assert report.rows_with_prior_value < report.rows
        assert report.leakage.ok

    def test_a_missing_input_table_is_a_keyerror_not_an_empty_table(self, tmp_path: object) -> None:
        from pathlib import Path

        from src.pipelines.features import build_features
        from src.storage.duckdb_store import DuckDBParquetStore

        store = DuckDBParquetStore(Path(str(tmp_path)) / "processed")
        with pytest.raises(KeyError):
            build_features(store)

    def test_a_leak_stops_the_run_and_writes_nothing(
        self, sample_store: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stage must refuse to write when the detector fires.

        Detection itself is covered by test_leakage.py and by
        ``test_a_leaky_table_is_caught_rather_than_assumed_impossible``; what is
        under test here is the wiring. A leaky table on disk is worse than no
        table, because the next stage cannot tell the difference.
        """
        from src.pipelines import features as stage
        from src.validation.report import (
            Finding,
            Severity,
            ValidationError,
            ValidationReport,
        )

        class FailingValidator:
            def validate(self, *_: object, **__: object) -> ValidationReport:
                report = ValidationReport()
                report.add(
                    Finding("leakage_feature_time", Severity.ERROR, "training_table", "boom")
                )
                return report

        monkeypatch.setattr(stage, "leakage_validator", lambda _columns: FailingValidator())

        with pytest.raises(ValidationError):
            stage.build_features(sample_store)  # type: ignore[arg-type]

        assert not sample_store.has_table(stage.TRAINING_TABLE)  # type: ignore[attr-defined]
