"""Where a player played, and how strong that context was.

The audit's third limitation: "20 goals in the Premier League is not 20 goals
in a weaker league", and the model could not tell the difference because
``competition_id`` and ``player_club_id`` were aggregated away in
``aggregate_appearances`` before anything looked at them.

Three families of context feature are built here.

**Competition identity and static attributes** — from ``competitions.csv``.
Confederation, competition type (domestic league, domestic cup, international
cup) and the size of the competition. These are properties of the competition,
not of any season, so they carry no time dimension and cannot leak.

**Club results strength** — from ``club_games.csv``. Points per game, goal
difference and league position, accumulated **up to the row's as-of date**
rather than over the whole season. It is the honest measure — a club's actual
results, not a reputation.

The as-of bound is not fussiness. This module previously argued that a whole-season
mean was safe "because the as-of date is the later of the season boundary and
the player's last appearance, so every result in that season is already
observed". That argument is wrong by up to thirty days. A season runs
August-July and is named for the August, so a match played on 20 July belongs
to it while sitting *after* the 1 July boundary that most as-of dates fall on:
5.56% of all fixtures are dated in July. Measured on the full panel, the
whole-season mean folded post-as-of matches into 15,126 rows (17.6%), and for
1,049 of them (1.22%) those matches were played after the *label* was set —
label leakage, in a numeric aggregate where no date-column check could see it.

Removing it costs nothing measurable (paired on the 2022 validation season,
EUR -10,497 for performance-only at t = -0.69, p = 0.49; EUR -1,684 at
p = 0.89 with prior value), which is the point: the leak was not buying
accuracy, only overstating provenance.

**Competition value level** — the one that needs care, and the reason this
module exists rather than a few extra lines in ``build.py``. The natural
measure of "how strong is this league" is the market value of the players in
it, and that is the target. Computing it from the current season would be
textbook target leakage: the label would be encoded into a feature under
another name.

So it is computed with a **strictly expanding window**: the value level for
competition *c* in season *s* uses only seasons *< s*. Season *s* itself never
contributes to its own feature. A competition appearing for the first time has
no history and gets a null, which the fitted imputer fills exactly as it does
for any other missing value. :func:`competition_strength` asserts the shift
rather than describing it, and ``tests/unit/test_context.py`` asserts it again
on a frame built to fail if the shift is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

MIN_ROWS_FOR_STRENGTH = 20
"""Below this a competition-season's mean is noise. Such rows contribute to the
expanding history but do not produce a strength value of their own."""

COMPETITION_ATTRIBUTES = ("competition_id", "type", "confederation", "country_name", "total_clubs")

CONTEXT_NUMERIC = (
    "club_points_per_game",
    "club_goal_difference_per_game",
    "club_league_position",
    "competition_value_level",
    "competition_tier_rank",
    "competitions_played",
    "continental_minutes_share",
)

CONTEXT_CATEGORICAL = (
    "primary_competition_type",
    "primary_confederation",
)


def _season_of(dates: pd.Series, start_month: int = 8) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    return (parsed.dt.year - (parsed.dt.month < start_month)).astype("Int64")


def player_competition_mix(
    appearances: pd.DataFrame, competitions: pd.DataFrame, *, start_month: int = 8
) -> pd.DataFrame:
    """One row per player-season describing *where* the minutes were played.

    The primary competition is the one carrying the most minutes, which is the
    honest answer to "which league is this player in" for someone who also
    played three cup ties and two continental qualifiers.
    """
    frame = appearances.loc[
        :, ["player_id", "date", "competition_id", "minutes_played", "player_club_id"]
    ].copy()
    frame["season"] = _season_of(frame["date"], start_month)
    frame = frame.dropna(subset=["season"]).astype({"season": "int64"})

    attrs = competitions.loc[:, list(COMPETITION_ATTRIBUTES)].rename(
        columns={"type": "competition_type"}
    )
    frame = frame.merge(attrs, on="competition_id", how="left")

    by_pair = frame.groupby(["player_id", "season"], sort=False)

    # Minutes-weighted primary competition, and the club that played most of them.
    def _dominant(column: str) -> pd.Series:
        totals = (
            frame.groupby(["player_id", "season", column], sort=False)["minutes_played"]
            .sum()
            .reset_index()
            .sort_values("minutes_played", ascending=False)
            .drop_duplicates(["player_id", "season"])
        )
        return totals.set_index(["player_id", "season"])[column]

    out = pd.DataFrame(
        {
            "competitions_played": by_pair["competition_id"].nunique(),
            "total_minutes_ctx": by_pair["minutes_played"].sum(),
        }
    )
    out["primary_competition_id"] = _dominant("competition_id")
    out["primary_club_id"] = _dominant("player_club_id")

    # Continental football is the clearest single signal of squad quality that
    # results alone do not carry: a Champions League group stage is a stronger
    # statement about a player than a domestic mid-table finish.
    continental = frame[frame["competition_type"].eq("international_cup")]
    cont_minutes = continental.groupby(["player_id", "season"])["minutes_played"].sum()
    out["continental_minutes_share"] = cont_minutes.reindex(out.index).fillna(0.0) / out[
        "total_minutes_ctx"
    ].clip(lower=1)

    out = out.reset_index()
    out = out.merge(
        attrs.rename(
            columns={
                "competition_id": "primary_competition_id",
                "competition_type": "primary_competition_type",
                "confederation": "primary_confederation",
                "country_name": "primary_country",
                "total_clubs": "primary_total_clubs",
            }
        ),
        on="primary_competition_id",
        how="left",
    )
    logger.info("competition mix for %d player-seasons", len(out))
    return out


def club_season_strength(club_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """A club's running record: one row per club-season-**match date**.

    Not one row per club-season. The caller joins this with
    :func:`pandas.merge_asof` on the player-season's as-of date, so a row is
    described by the club's form *at the moment that row's evidence closed* and
    never by matches played afterwards. See this module's docstring for the
    measured size of the leak the whole-season version carried.

    Returning the running record rather than accepting the as-of dates here
    keeps the aggregation independent of which frame is being enriched — the
    labelled build and the current-season build hand it the same club history
    and differ only in where they cut it.

    ``club_matches`` is the count to date, which is also the availability
    denominator :mod:`src.feature_engineering.performance` needs: the player's
    appearances are counted to the same as-of date, so numerator and
    denominator now close at the same moment.

    ``is_win`` is 1/0 in this source and does not distinguish a draw from a
    defeat, so points are reconstructed from the goals instead.
    """
    joined = club_games.merge(
        games.loc[:, ["game_id", "date", "season", "competition_type"]], on="game_id", how="left"
    ).dropna(subset=["season", "date"])
    joined = joined.astype({"season": "int64"})
    joined["date"] = pd.to_datetime(joined["date"], errors="coerce")
    joined = joined.dropna(subset=["date"])

    goals_for = pd.to_numeric(joined["own_goals"], errors="coerce")
    goals_against = pd.to_numeric(joined["opponent_goals"], errors="coerce")
    joined["points"] = np.where(
        goals_for > goals_against, 3.0, np.where(goals_for == goals_against, 1.0, 0.0)
    )
    joined["goal_difference"] = goals_for - goals_against
    joined["position"] = pd.to_numeric(joined["own_position"], errors="coerce")

    joined = joined.sort_values(["club_id", "season", "date"])
    by_club_season = joined.groupby(["club_id", "season"], sort=False)

    # Cumulative rather than aggregate: match n's row carries the record over
    # matches 1..n. Two matches on the same date both appear; merge_asof takes
    # the last one at or before the as-of date, which is the complete record
    # for that day.
    joined["club_matches"] = by_club_season.cumcount() + 1
    joined["club_points_per_game"] = by_club_season["points"].cumsum() / joined["club_matches"]
    joined["club_goal_difference_per_game"] = (
        by_club_season["goal_difference"].cumsum() / joined["club_matches"]
    )
    joined["club_league_position"] = by_club_season["position"].transform(
        lambda s: s.expanding().mean()
    )

    running = joined.loc[
        :,
        [
            "club_id",
            "season",
            "date",
            "club_matches",
            "club_points_per_game",
            "club_goal_difference_per_game",
            "club_league_position",
        ],
    ].reset_index(drop=True)
    logger.info(
        "club running strength: %d match rows over %d club-seasons",
        len(running),
        by_club_season.ngroups,
    )
    return running


def competition_strength(labelled: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    """Value level per competition-season, from **strictly earlier** seasons only.

    This is a target encoding, which is the most reliable way to leak a label
    into a feature. The defence is arithmetic rather than careful: the
    per-season mean is computed, then ``shift(1)`` moves it one season forward
    and ``expanding().mean()`` accumulates only what came before. Season *s*
    contributes to *s+1* and never to itself, so a row's own label is not in
    its own feature by construction.

    Returns one row per (competition, season) with the level to attach to that
    season. Competitions in their first observed season get NaN, which the
    fitted imputer handles like any other missing value.
    """
    frame = labelled.loc[:, ["primary_competition_id", "season", target_column]].dropna()
    per_season = (
        frame.groupby(["primary_competition_id", "season"], as_index=False)
        .agg(
            level=(target_column, lambda s: float(np.log1p(s).mean())), rows=(target_column, "size")
        )
        .sort_values(["primary_competition_id", "season"])
    )
    # Seasons too thin to mean anything still count toward history; they just
    # do not get to define a level on their own.
    per_season.loc[per_season["rows"] < MIN_ROWS_FOR_STRENGTH, "level"] = np.nan

    grouped = per_season.groupby("primary_competition_id", sort=False)["level"]
    per_season["competition_value_level"] = grouped.transform(
        lambda s: s.shift(1).expanding().mean()
    )

    out = per_season.loc[
        :, ["primary_competition_id", "season", "competition_value_level"]
    ].reset_index(drop=True)

    # Rank within season, so the feature is "how strong relative to the rest of
    # the panel this year" rather than an absolute figure that drifts with
    # football's general inflation.
    out["competition_tier_rank"] = out.groupby("season")["competition_value_level"].rank(
        pct=True, ascending=True
    )
    logger.info(
        "competition strength for %d competition-seasons (%d with history)",
        len(out),
        int(out["competition_value_level"].notna().sum()),
    )
    return out


CLUB_RECORD_DATE_COLUMN = "club_record_date"
"""Date of the most recent club match folded into this row's club columns.

Carried into the training table on purpose. The club record is joined with
``merge_asof``, and a merge direction is an argument rather than a fact; this
column turns it into one a check can read.
"""

CLUB_RUNNING_COLUMNS = (
    "club_matches",
    "club_points_per_game",
    "club_goal_difference_per_game",
    "club_league_position",
)
"""What :func:`club_season_strength` contributes, joined as of the row's date."""


def attach_context(
    table: pd.DataFrame,
    *,
    competition_mix: pd.DataFrame,
    club_strength: pd.DataFrame,
    competition_levels: pd.DataFrame,
    as_of_column: str = "as_of_date",
) -> pd.DataFrame:
    """Join every context feature onto the player-season table.

    The club record joins with :func:`pandas.merge_asof`, not an equality
    merge: :func:`club_season_strength` returns a running record and the row is
    entitled to the part of it that had happened by ``as_of_column``. Every
    other join is on keys that carry no within-season time dimension.
    """
    out = table.merge(
        competition_mix.drop(columns=["total_minutes_ctx"], errors="ignore"),
        on=["player_id", "season"],
        how="left",
    )
    out = _attach_club_record(out, club_strength, as_of_column=as_of_column)
    return out.merge(competition_levels, on=["primary_competition_id", "season"], how="left")


def _attach_club_record(
    table: pd.DataFrame, club_strength: pd.DataFrame, *, as_of_column: str
) -> pd.DataFrame:
    """The club's record as it stood on the row's as-of date.

    ``merge_asof`` requires both sides sorted on the join key and returns a
    fresh index, so the original index is carried through explicitly rather
    than assumed — the caller's frame is not always a clean RangeIndex.
    """
    if as_of_column not in table.columns or club_strength.empty:
        for column in CLUB_RUNNING_COLUMNS:
            table[column] = np.nan
        return table

    right = club_strength.rename(
        columns={"club_id": "primary_club_id", "date": CLUB_RECORD_DATE_COLUMN}
    ).sort_values(CLUB_RECORD_DATE_COLUMN)
    left = table.sort_values(as_of_column)
    merged = pd.merge_asof(
        left,
        right,
        left_on=as_of_column,
        right_on=CLUB_RECORD_DATE_COLUMN,
        by=["primary_club_id", "season"],
        direction="backward",
        allow_exact_matches=True,
    )
    merged.index = left.index
    # Kept, not dropped. It is the provenance of every club column on the row —
    # the date of the last match folded into them — so leaving it in the frame
    # is what lets `check_no_future_dates` audit this join by value instead of
    # anyone having to trust the merge direction. See that check's docstring.
    return merged.reindex(table.index)


__all__ = [
    "CLUB_RECORD_DATE_COLUMN",
    "CLUB_RUNNING_COLUMNS",
    "CONTEXT_CATEGORICAL",
    "CONTEXT_NUMERIC",
    "MIN_ROWS_FOR_STRENGTH",
    "attach_context",
    "club_season_strength",
    "competition_strength",
    "player_competition_mix",
]
