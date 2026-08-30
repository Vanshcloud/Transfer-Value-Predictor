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

from src.feature_engineering.context import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    attach_context,
    club_season_strength,
    competition_strength,
    player_competition_mix,
)
from src.feature_engineering.performance import (
    PERFORMANCE_NUMERIC,
    attach_performance,
    match_level_features,
    squad_match_counts,
    squad_role,
)
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
    (
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
    + PERFORMANCE_NUMERIC
    + CONTEXT_NUMERIC
)

CATEGORICAL_FEATURES = (
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
) + CONTEXT_CATEGORICAL

LABEL_HORIZON_COLUMN = "label_horizon_days"
"""Days between the as-of date and the valuation used as the label.

Recorded on every row, and deliberately **not** a feature.

Widening the label window from 120 to 365 days spreads the label over a year,
and the obvious worry is that the model is then averaging over an unspecified
horizon. The intended fix was to hand it the horizon so it predicts a specified
one. Measured, that bought nothing: R^2 0.760 with the feature against 0.762
without it, on the same rows and the same split. A feature that moves the third
decimal the wrong way is not worth the interface it would add — every caller
would have to supply a horizon, and the model would not listen.

So it stays as a column. It is what makes the label's provenance auditable, it
is what :func:`~src.pipelines.tune.season_weights` could down-weight on, and it
is the evidence for the claim that widening the window did no harm.
"""

NON_NEGATIVE_FEATURES = frozenset(
    {
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
        "years_since_debut",
        "seasons_observed",
    }
)
"""Features that cannot be negative by definition — counts, durations, rates
and physical measurements.

Listed here rather than in the API layer because it is a property of what the
feature *means*, and the same rule has to hold for a batch job or a CLI that
never touches HTTP. A negative goal count is not an unusual player; it is a
caller bug, and answering it with a confident number is the failure mode this
project spends most of its effort avoiding.

``prev_log_market_value_in_eur`` and ``prev_value_age_days`` are deliberately
absent: the first is a log of a value that can legitimately sit below 1, and
the second is a signed staleness in days.
"""

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
    tolerance_days: int = 365,
) -> pd.DataFrame:
    """Label each player-season with the first valuation on or after its as-of date.

    ``direction="forward"`` is the whole point: the label must be set *after*
    the features were observed. The tolerance bounds how long we will wait.

    That bound was 120 days and cost 61% of the panel. Transfermarkt revalues
    in batches, and the largest of them lands in the winter window — a window
    that closes on 29 October never sees it. Measured over the full panel:

        120 days -> 39.0% of player-seasons labelled   (36,902)
        180 days -> 74.7%                              (70,735)
        365 days -> 90.8%                              (86,016)

    Widening cannot introduce leakage: every label is still strictly forward of
    the as-of date, which is the only property that matters for correctness.
    What it does change is how far ahead the label sits, and a label 300 days
    out is partly about the *next* season. That distance is recorded on every
    row as ``label_horizon_days`` so the claim is auditable — and it was tried
    as a feature and dropped, because it moved test R^2 from 0.762 to 0.760.
    The wider window costs less than the theory predicted.

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

    labelled = merged.dropna(subset=[TARGET_COLUMN, LABEL_TIME_COLUMN]).copy()
    labelled[LABEL_HORIZON_COLUMN] = (
        labelled[LABEL_TIME_COLUMN] - labelled[AS_OF_COLUMN]
    ).dt.days.astype("float64")
    logger.info(
        "labelled %d of %d player-seasons (%.1f%%) within %d days; median horizon %.0f days",
        len(labelled),
        len(merged),
        100.0 * len(labelled) / max(len(merged), 1),
        tolerance_days,
        float(labelled[LABEL_HORIZON_COLUMN].median()),
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

    The lag is genuine, and is enforced rather than argued. Ordering by season
    does not by itself guarantee that the previous label predates this row's
    as-of date — with a 365-day label window, season s can be labelled after
    season s+1 has begun. Those rows get no prior value at all.
    """
    ordered = frame.sort_values(["player_id", "season"])
    by_player = ordered.groupby("player_id", sort=False)

    prev_label_date = by_player[LABEL_TIME_COLUMN].shift(1)

    # A previous season's label is only usable if it had actually been
    # published by the time this season's evidence closed. Ordering by season
    # is not enough to guarantee that once the label window is a year wide:
    # season s can be labelled in the June after season s+1 has already begun,
    # in which case s+1's "prior" value is not prior at all.
    #
    # Measured on the full panel when the window moved from 120 to 365 days,
    # this affected 22 rows of 61,555 — 0.03%. Small enough to miss by eye and
    # exactly the kind of thing that survives because it is small, so it is
    # nulled here and asserted by
    # `src.validation.leakage.check_lagged_values_precede_features`.
    # Strictly before, not on-or-before. A valuation published the same day the
    # feature window closes may already reflect that day's match, and "prior"
    # should not need an argument about intraday ordering to be true.
    knowable = prev_label_date.notna() & (prev_label_date < ordered[AS_OF_COLUMN])

    ordered["prev_market_value_in_eur"] = by_player[TARGET_COLUMN].shift(1).where(knowable)
    ordered["prev_log_market_value_in_eur"] = np.log1p(ordered["prev_market_value_in_eur"])
    # Measured from the as-of date, not from the current label: staleness is a
    # property of what was known when the features closed, and the label date
    # is not known then.
    ordered["prev_value_age_days"] = (
        (ordered[AS_OF_COLUMN] - prev_label_date).dt.days.where(knowable).astype("float64")
    )

    # Counts only the rows before this one, so it is a running career depth
    # rather than a total that would need the player's whole future to compute.
    # Capped for the same censoring reason as years_since_debut: the count a
    # season can reach is bounded by how long the dataset has existed.
    ordered["seasons_observed"] = by_player.cumcount().clip(upper=int(CAREER_CENSORING_CEILING))

    return ordered.reset_index(drop=True)


def _mix_for_levels(
    table: pd.DataFrame,
    appearances: pd.DataFrame,
    competitions: pd.DataFrame,
    season_start_month: int,
) -> pd.DataFrame:
    """Labelled rows tagged with their primary competition, for the strength pass.

    :func:`~src.feature_engineering.context.competition_strength` needs to know
    which competition each *labelled* row belongs to before it can average
    values per competition-season. It gets the labelled table rather than the
    raw appearances so that the history it accumulates is the history the model
    is actually trained on.
    """
    mix = player_competition_mix(appearances, competitions, start_month=season_start_month)
    # Dropped before the merge, not renamed after it: an already-enriched frame
    # (the training table, when the unlabelled build passes it as history)
    # carries this column, and pandas would otherwise silently produce
    # `primary_competition_id_x` and `_y` and leave neither under the name the
    # next step looks for.
    return table.drop(columns=["primary_competition_id"], errors="ignore").merge(
        mix.loc[:, ["player_id", "season", "primary_competition_id"]],
        on=["player_id", "season"],
        how="left",
    )


def _ensure_feature_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every declared feature exists, as a column of nulls if need be.

    The context and performance joins are optional — the sample-data path runs
    without them. Without this, ``FEATURE_COLUMNS`` would name columns the frame
    does not have and the failure would surface deep inside a ColumnTransformer
    as a KeyError about a column nobody mentioned. A declared-but-absent feature
    is missing data, which the fitted imputer already knows how to handle, so it
    is represented as exactly that.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in table.columns]
    if missing:
        logger.info(
            "%d context/performance feature(s) unavailable in this build: %s",
            len(missing),
            ", ".join(missing),
        )
        for column in missing:
            # np.nan and None, never pd.NA. scikit-learn's SimpleImputer tests
            # missingness with `X != X`, and pd.NA raises
            # "boolean value of NA is ambiguous" from inside a ColumnTransformer
            # — a failure whose traceback names neither the column nor the
            # feature that caused it.
            if column in CATEGORICAL_FEATURES:
                table[column] = pd.Series(None, index=table.index, dtype="object")
            else:
                table[column] = pd.Series(np.nan, index=table.index, dtype="float64")
    return table


def _enrich(
    table: pd.DataFrame,
    *,
    appearances: pd.DataFrame,
    competitions: pd.DataFrame | None,
    games: pd.DataFrame | None,
    club_games: pd.DataFrame | None,
    lineups: pd.DataFrame | None,
    season_start_month: int,
    levels_from: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach context and richer performance to a player-season frame.

    Shared by the labelled and unlabelled builds so a prediction row is
    assembled by exactly the code that assembled the rows the model trained on.
    Two implementations of a feature definition is the standard way training and
    serving drift apart, and this project already refuses that trade in
    ``ModelArtifact`` by keeping the fitted preprocessing inside the pipeline.

    Optional because the committed sample data carries only the three required
    tables. Absent enrichment leaves those features null, which the fitted
    imputer fills exactly as it does any other missing value.

    ``levels_from`` supplies the labelled history used to compute competition
    strength. The unlabelled build passes the *training* table here: a season
    with no labels can contribute nothing to a value level, and must instead
    inherit the level accumulated from every season before it.
    """
    if competitions is None or games is None or club_games is None:
        return table

    history = table if levels_from is None else levels_from
    enriched = attach_context(
        table,
        competition_mix=player_competition_mix(
            appearances, competitions, start_month=season_start_month
        ),
        club_strength=club_season_strength(club_games, games),
        competition_levels=competition_strength(
            _mix_for_levels(history, appearances, competitions, season_start_month),
            target_column=TARGET_COLUMN,
        ),
    )
    return attach_performance(
        enriched,
        match_features=match_level_features(appearances, start_month=season_start_month),
        role=(
            squad_role(lineups, start_month=season_start_month)
            if lineups is not None
            else pd.DataFrame(columns=["player_id", "season"])
        ),
        squad_matches=squad_match_counts(club_games, games),
    )


def build_training_table(
    players: pd.DataFrame,
    valuations: pd.DataFrame,
    appearances: pd.DataFrame,
    *,
    competitions: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
    club_games: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    season_start_month: int = 8,
    label_tolerance_days: int = 365,
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

    table = _enrich(
        table,
        appearances=appearances,
        competitions=competitions,
        games=games,
        club_games=club_games,
        lineups=lineups,
        season_start_month=season_start_month,
        levels_from=None,
    )
    table = _ensure_feature_columns(table)

    logger.info(
        "training table: %d rows, %d players, seasons %d-%d",
        len(table),
        table["player_id"].nunique(),
        table["season"].min(),
        table["season"].max(),
    )
    return table


def build_current_season_table(
    players: pd.DataFrame,
    valuations: pd.DataFrame,
    appearances: pd.DataFrame,
    training_table: pd.DataFrame,
    *,
    competitions: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
    club_games: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    season_start_month: int = 8,
) -> pd.DataFrame:
    """Feature rows for seasons that have no label yet.

    The audit's sixth limitation: the freshest predictable season was already
    a year old, because a row cannot enter the training table until a valuation
    exists to label it, and that valuation only appears after the season ends.
    So the most recent season — the one anybody actually wants a number for —
    was invisible to the service.

    It is invisible for *training*, correctly and permanently. It does not have
    to be invisible for *prediction*: the features are complete the moment the
    matches are played, and the model needs no label to score a row. This
    builds exactly those rows.

    Two properties make it safe:

    * These rows are never returned to the training pipeline. They carry no
      target, so a fit would fail rather than quietly learn from a null.
    * ``prev_market_value_in_eur`` comes from the *labelled* history, which is
      strictly earlier, so the prior-value variant works here on genuinely past
      information.

    A partially-played season is not a defect either. Half a season of matches
    is half a season of evidence, and the model prices it as such —
    ``appearances``, ``squad_match_share`` and ``months_active`` all fall,
    which is the honest signal that there is less to go on.

    Returns rows for every season strictly after the last labelled one.
    """
    player_seasons = aggregate_appearances(appearances, start_month=season_start_month)
    last_labelled = int(training_table["season"].max())
    current = player_seasons[player_seasons["season"] > last_labelled].copy()
    if current.empty:
        logger.info("no unlabelled seasons after %d", last_labelled)
        return current

    # The label window is what these rows lack, so the columns it would have
    # produced are declared absent rather than left for a downstream KeyError.
    current[LABEL_TIME_COLUMN] = pd.NaT
    current[TARGET_COLUMN] = np.nan
    current[LABEL_HORIZON_COLUMN] = np.nan

    with_attributes = attach_player_attributes(current, players)
    table = add_derived_features(with_attributes)

    # The prior value is the last valuation published *strictly before* this
    # season's as-of date. Taken from the raw valuations rather than from the
    # previous labelled row, because a player may have been revalued since —
    # and a fresher prior is a better one, as `prev_value_age_days` records.
    # ``direction="backward"`` is what makes it strictly past information.
    prior = (
        valuations.loc[:, ["player_id", "date", TARGET_COLUMN]]
        .assign(date=pd.to_datetime(valuations["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .rename(columns={"date": "_prev_label_date", TARGET_COLUMN: "prev_market_value_in_eur"})
    )
    table = pd.merge_asof(
        table.sort_values(AS_OF_COLUMN),
        prior,
        left_on=AS_OF_COLUMN,
        right_on="_prev_label_date",
        by="player_id",
        direction="backward",
        allow_exact_matches=False,
    )
    table["prev_log_market_value_in_eur"] = np.log1p(table["prev_market_value_in_eur"])
    table["prev_value_age_days"] = (table[AS_OF_COLUMN] - table["_prev_label_date"]).dt.days.astype(
        "float64"
    )

    # Career depth is how many labelled seasons this player already has, which
    # is what the same column counts during training.
    depth = training_table.groupby("player_id").size()
    table["seasons_observed"] = (
        table["player_id"].map(depth).fillna(0).clip(upper=CAREER_CENSORING_CEILING)
    )

    table = _enrich(
        table,
        appearances=appearances,
        competitions=competitions,
        games=games,
        club_games=club_games,
        lineups=lineups,
        season_start_month=season_start_month,
        levels_from=training_table,
    )
    table = _ensure_feature_columns(table)
    table = table.drop(columns=["_prev_label_date"], errors="ignore")

    logger.info(
        "current-season table: %d rows, %d players, seasons %d-%d (unlabelled)",
        len(table),
        table["player_id"].nunique(),
        table["season"].min(),
        table["season"].max(),
    )
    return table.reset_index(drop=True)


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
