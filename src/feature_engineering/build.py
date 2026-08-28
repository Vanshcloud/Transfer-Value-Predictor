"""The training table: one row per player-season, labelled with a future valuation.

This is the join the whole project rests on. Three raw tables go in and one
modelling frame comes out, built in an order chosen so that no step can see
data that did not exist yet:

1. Appearances are stamped with a season (August-July) and aggregated to one
   row per player-season. The latest appearance date survives the aggregation
   as ``last_appearance_date``.
2. Each player-season gets an **as-of date**: the later of the 1 July season
   boundary and that row's last appearance. The label is then the first
   valuation recorded at or after it, found with :func:`pandas.merge_asof` and
   a 120-day tolerance. Valuations are an irregular on-change event series, not
   a schedule, so an equality merge finds almost nothing and a backward merge
   finds the future.
3. Player attributes are joined last, and only the ones that are properties of
   the person rather than of today: birth date, position, foot, height,
   nationality. ``contract_expiration_date``, ``market_value_in_eur`` and the
   ``current_club_*`` family are *current state* — attaching them to a 2015 row
   states a fact from 2026. They never enter this frame.

The prior-season value is computed here but kept out of the default feature
list. It moves R^2 from 0.45 to 0.83 and drowns every other signal, so the
project ships two variants from this one code path — see :func:`select_variant`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "market_value_in_eur"
"""The label, in EUR. Models train on ``log1p`` of it (skew 8.70 -> 0.43) and
report in EUR; that transform belongs to the model, not to the table."""

FEATURE_TIME_COLUMN = "last_appearance_date"
"""The most recent moment any feature in the row was observed."""

LABEL_TIME_COLUMN = "label_date"
"""When the label was published. Must never precede the feature time."""

AS_OF_COLUMN = "as_of_date"
"""The boundary between the feature window and the label window."""

# Appearance coverage starts 2012-07-03, so "years since debut" is really
# "years since the dataset started watching": its maximum runs 0, 1, 2 ... 13
# in lockstep with the season, which reconstructs the calendar variable this
# project deliberately excluded. Capping restores a common support — every
# season from 2021 on can express the full 0-10 range, so a test-set row never
# carries a value no training row could hold. Seasons before that stay
# left-censored, which adds noise inside the training range but cannot produce
# extrapolation beyond it.
# ponytail: a fixed ceiling, because the censoring is a property of this
# dataset's start date. Re-derive it if the coverage window changes.
CAREER_CENSORING_CEILING = 10.0

# A per-90 rate divides by minutes. One goal in a three-minute substitute
# appearance is 30 goals per 90 without a floor, and that row then outweighs a
# whole season of a real striker. The floor is one full match.
MINUTES_FLOOR = 90.0

# Player attributes that describe the person, not the present. Everything else
# in players.csv is either current state (banned outright, see
# src/validation/leakage.CURRENT_STATE_COLUMNS) or an identifier.
PLAYER_ATTRIBUTES = (
    "player_id",
    "date_of_birth",
    "position",
    "sub_position",
    "foot",
    "height_in_cm",
    "country_of_citizenship",
)

NUMERIC_FEATURES = (
    "age",
    "age_squared",
    "appearances",
    "goals",
    "assists",
    "minutes_played",
    "yellow_cards",
    "red_cards",
    "goals_per_90",
    "assists_per_90",
    "cards_per_90",
    "minutes_per_appearance",
    "height_in_cm",
    # Where the player is in a career, not where the season is in the calendar.
    # Both answer "how established is this player" without ever encoding
    # "2024 players are worth more than 2022 players".
    "years_since_debut",
    "seasons_observed",
)

CATEGORICAL_FEATURES = (
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
)

PRIOR_VALUE_FEATURES = ("prev_log_market_value_in_eur", "prev_value_age_days")
"""The lagged label in log space, and how stale it was when the current label
was set.

Log, not EUR, because the *target* is modelled as log1p: relating log(value_t)
to a raw-EUR value_(t-1) is a misspecified linear model. Raw, the column has
skew 4.53 and its largest value sits 17 standard deviations out, so a linear
coefficient on that row produces a large prediction in log space that expm1
turns into billions — measured R^2 of -1.8 million for Ridge. Under log1p the
skew is 0.10. Tree models are invariant to monotone transforms and never saw
the problem, which is precisely why it would have shipped.
The raw EUR column stays in the table for readability and for variant
selection; it is simply not what the model is handed.

Named with the ``prev_`` prefix because that is what the leakage detector
accepts as proof a value is lagged on purpose
(src/validation/leakage.check_target_absent_from_features). The staleness
column is not decoration: a prior value can be one season old or four, and a
four-year-old valuation deserves less weight than last summer's."""

# `season` is deliberately not a feature. It is the split key, and a model that
# learns "later season -> higher value" extrapolates a trend straight off the
# end of its training range the moment it meets the test years.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def assign_season(dates: pd.Series, start_month: int = 8) -> pd.Series:
    """Map match dates to the season they belong to.

    A season is named for the calendar year it starts in, so 2012-05-19 belongs
    to season 2011. Returns a nullable integer, because an unparseable date has
    no season and must not silently become one.
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    return (parsed.dt.year - (parsed.dt.month < start_month)).astype("Int64")


def season_boundary_dates(seasons: pd.Series, start_month: int = 8) -> pd.Series:
    """The conventional end of each season: 1 July for an August start.

    Football's contract year turns over on 1 July, and it is also the month
    Transfermarkt publishes its largest batch of revaluations, so it is the
    natural line between "what the player did" and "what he is now worth".
    """
    month = start_month - 1
    years = seasons.astype("int64") + 1
    if month == 0:  # a January-start season ends the December before
        month, years = 12, years - 1
    return pd.to_datetime({"year": years, "month": month, "day": 1})


def aggregate_appearances(appearances: pd.DataFrame, *, start_month: int = 8) -> pd.DataFrame:
    """Collapse per-match rows into one row per player-season.

    The as-of date is ``max(season boundary, last appearance)`` rather than the
    boundary alone. Roughly a fifth of player-seasons are still being played in
    July — summer-calendar leagues, continental tournaments — and anchoring
    those on 1 July would let a valuation published mid-tournament label a row
    whose features run past it. Taking the later of the two makes
    ``last_appearance_date <= label_date`` true *by construction* rather than
    by hope; the alternative, truncating the season at 1 July, silently drops
    2,772 player-seasons and most of MLS.
    """
    frame = appearances.assign(date=pd.to_datetime(appearances["date"], errors="coerce"))
    frame["season"] = assign_season(frame["date"], start_month)
    frame = frame.dropna(subset=["season", "date"])

    grouped = (
        frame.groupby(["player_id", "season"], as_index=False)
        .agg(
            appearances=("appearance_id", "count"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            minutes_played=("minutes_played", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            last_appearance_date=("date", "max"),
        )
        .astype({"season": "int64"})
    )

    boundary = season_boundary_dates(grouped["season"], start_month)
    grouped[AS_OF_COLUMN] = boundary.where(
        boundary >= grouped[FEATURE_TIME_COLUMN], grouped[FEATURE_TIME_COLUMN]
    )

    # Career stage, not calendar position. The debut is the player's first
    # appearance anywhere in the dataset, which is necessarily at or before
    # their first appearance in the season being described — so it is knowable
    # at the as-of date, and a 21-year-old six seasons into a career is a
    # different proposition from a 21-year-old in his first.
    debut = frame.groupby("player_id")["date"].min()
    grouped["years_since_debut"] = (
        (grouped[AS_OF_COLUMN] - grouped["player_id"].map(debut)).dt.days / 365.25
    ).clip(upper=CAREER_CENSORING_CEILING)

    logger.info("aggregated %d appearances into %d player-seasons", len(appearances), len(grouped))
    return grouped


def attach_label(
    player_seasons: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    tolerance_days: int = 120,
) -> pd.DataFrame:
    """Label each player-season with the first valuation on or after its as-of date.

    ``direction="forward"`` is the whole point: the label must be set *after*
    the features were observed. The tolerance bounds how long we will wait —
    beyond it the valuation reflects a later season's form, so the row is
    dropped rather than mislabelled.

    ``pd.Timedelta(120, "D")`` rather than ``pd.Timedelta("120D")`` or
    ``days=120``: both of the latter emit a DeprecationWarning under
    numpy 2.5, and the test suite runs with ``-W error::DeprecationWarning``.
    """
    right = (
        valuations.loc[:, ["player_id", "date", TARGET_COLUMN]]
        .assign(date=pd.to_datetime(valuations["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .rename(columns={"date": LABEL_TIME_COLUMN})
    )

    merged = pd.merge_asof(
        player_seasons.sort_values(AS_OF_COLUMN),
        right,
        left_on=AS_OF_COLUMN,
        right_on=LABEL_TIME_COLUMN,
        by="player_id",
        direction="forward",
        tolerance=pd.Timedelta(tolerance_days, "D"),
    )

    labelled = merged.dropna(subset=[TARGET_COLUMN, LABEL_TIME_COLUMN])
    logger.info(
        "labelled %d of %d player-seasons within %d days of the as-of date",
        len(labelled),
        len(merged),
        tolerance_days,
    )
    return labelled.reset_index(drop=True)


def attach_player_attributes(frame: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Join the player attributes that are not current state, and derive age."""
    attributes = players.loc[:, list(PLAYER_ATTRIBUTES)].copy()
    attributes["date_of_birth"] = pd.to_datetime(attributes["date_of_birth"], errors="coerce")

    joined = frame.merge(attributes, on="player_id", how="left", validate="many_to_one")
    joined["age"] = (joined[AS_OF_COLUMN] - joined["date_of_birth"]).dt.days / 365.25

    # Age is a core feature and cannot be imputed from anything else in the
    # table, so a row without a birth date is not modellable.
    missing_age = int(joined["age"].isna().sum())
    if missing_age:
        logger.warning("dropping %d row(s) with no date of birth", missing_age)
    return joined.dropna(subset=["age"]).reset_index(drop=True)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rates and curve terms — everything that is a function of existing columns."""
    per_90 = 90.0 / frame["minutes_played"].clip(lower=MINUTES_FLOOR)

    return frame.assign(
        goals_per_90=frame["goals"] * per_90,
        assists_per_90=frame["assists"] * per_90,
        cards_per_90=(frame["yellow_cards"] + frame["red_cards"]) * per_90,
        minutes_per_appearance=frame["minutes_played"] / frame["appearances"].clip(lower=1),
        # Market value peaks in the mid-twenties and falls away either side, so
        # the model needs a quadratic term to bend; a linear age can only slope.
        age_squared=frame["age"] ** 2,
    )


def add_career_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach everything that depends on the player's earlier rows.

    The lagged prior value, how stale it is, and how many earlier seasons this
    player already has in the table. All three look strictly backwards.

    "Previous" means the most recent earlier season in this table, not
    necessarily the season before: an injury year or a spell outside the
    covered leagues leaves a gap, and a three-year-old valuation is still
    informative as long as the model is told how old it is. That is what
    ``prev_value_age_days`` carries.

    The lag is genuine. Seasons are ordered, so the previous row's label was
    published before this row's as-of date; the test suite asserts it rather
    than trusting the argument.
    """
    ordered = frame.sort_values(["player_id", "season"])
    by_player = ordered.groupby("player_id", sort=False)

    ordered["prev_market_value_in_eur"] = by_player[TARGET_COLUMN].shift(1)
    ordered["prev_log_market_value_in_eur"] = np.log1p(ordered["prev_market_value_in_eur"])
    prev_label_date = by_player[LABEL_TIME_COLUMN].shift(1)
    ordered["prev_value_age_days"] = (ordered[LABEL_TIME_COLUMN] - prev_label_date).dt.days

    # Counts only the rows before this one, so it is a running career depth
    # rather than a total that would need the player's whole future to compute.
    # Capped for the same censoring reason as years_since_debut: the count a
    # season can reach is bounded by how long the dataset has existed.
    ordered["seasons_observed"] = by_player.cumcount().clip(upper=int(CAREER_CENSORING_CEILING))

    return ordered.reset_index(drop=True)


def build_training_table(
    players: pd.DataFrame,
    valuations: pd.DataFrame,
    appearances: pd.DataFrame,
    *,
    season_start_month: int = 8,
    label_tolerance_days: int = 120,
) -> pd.DataFrame:
    """Build the full player-season table, prior-value column included.

    One table is built, not two. ``prev_market_value_in_eur`` is null for a
    player's first covered season and populated afterwards, so the two model
    variants are a row filter and a column list away — see
    :func:`select_variant` — rather than a second pipeline free to drift from
    this one.
    """
    player_seasons = aggregate_appearances(appearances, start_month=season_start_month)
    labelled = attach_label(player_seasons, valuations, tolerance_days=label_tolerance_days)
    with_attributes = attach_player_attributes(labelled, players)
    table = add_career_history(add_derived_features(with_attributes))

    logger.info(
        "training table: %d rows, %d players, seasons %d-%d",
        len(table),
        table["player_id"].nunique(),
        table["season"].min(),
        table["season"].max(),
    )
    return table


def select_variant(
    table: pd.DataFrame, *, include_prior_value: bool
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Split the one table into one of the two shipped model variants.

    ``include_prior_value=False`` is the *performance-only* model: every row,
    and scouting or undervaluation are questions it can actually answer.
    ``True`` is the *accurate* model: it keeps only rows with a real prior
    season (roughly half) and predicts how a known value will move.

    Publishing only the second would be technically true and practically
    useless, which is why both come out of here.
    """
    if not include_prior_value:
        return table, FEATURE_COLUMNS

    with_prior = table.dropna(subset=["prev_market_value_in_eur"]).reset_index(drop=True)
    return with_prior, FEATURE_COLUMNS + PRIOR_VALUE_FEATURES


def null_rates(table: pd.DataFrame, columns: tuple[str, ...] = FEATURE_COLUMNS) -> pd.Series:
    """Fraction of nulls per feature column, for the verification report."""
    return table.loc[:, list(columns)].isna().mean()


__all__ = [
    "AS_OF_COLUMN",
    "CAREER_CENSORING_CEILING",
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "FEATURE_TIME_COLUMN",
    "LABEL_TIME_COLUMN",
    "NUMERIC_FEATURES",
    "PRIOR_VALUE_FEATURES",
    "TARGET_COLUMN",
    "add_derived_features",
    "add_career_history",
    "aggregate_appearances",
    "assign_season",
    "attach_label",
    "attach_player_attributes",
    "build_training_table",
    "null_rates",
    "season_boundary_dates",
    "select_variant",
]
