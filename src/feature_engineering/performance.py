"""Richer per-season performance, from files the project already downloads.

The audit's second limitation: the model had five counting statistics — matches,
minutes, goals, assists, cards — and nothing else. That is a thin basis for
valuing a footballer, and it was thin by omission rather than necessity: three
of the ten files in the same Kaggle download carry squad role, team results and
match context.

What is added, and why each earns its place:

**Role** — ``starts`` and ``substitute_appearances`` from ``game_lineups.csv``.
A player with 2,000 minutes across 25 starts is a different proposition from one
with 2,000 minutes across 40 appearances, and total minutes cannot tell them
apart. ``captain_share`` is the cheapest available proxy for standing within a
squad.

**Availability** — ``squad_match_share``: the player's matches divided by the
matches his club actually played that season. Thirty appearances means
something different in a 34-game league than in a 60-game European campaign,
and an injured season shows up here as a low share rather than as an
indistinguishable low minute count.

**Consistency** — ``full_match_share`` and ``minutes_variability``. Being
picked every week is information; being picked and then withdrawn on the hour
is different information.

**Trajectory** — ``second_half_goal_share``. A player who scored fifteen after
Christmas is priced differently from one who scored fifteen before it, and the
label is set in the summer that follows.

**Versatility** — ``positions_played``, distinct starting positions.

Everything here is computed from events inside the season being described, so
all of it is observable at the as-of date. Nothing reads a valuation.

``game_lineups.csv`` begins 2013-07-02, so role and versatility features are
null for seasons 2011 and 2012. They are left null rather than zero-filled:
the fitted imputer treats them as missing, which is true, whereas a zero would
assert that nobody started a match in 2012.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

FULL_MATCH_MINUTES = 60
"""What counts as a real involvement rather than a cameo. Sixty minutes is the
conventional line and matches the per-90 floor already used elsewhere."""

PERFORMANCE_NUMERIC = (
    "goal_contributions",
    "contributions_per_90",
    "starts",
    "substitute_appearances",
    "start_share",
    "captain_share",
    "squad_match_share",
    "full_match_share",
    "minutes_variability",
    "scoring_match_share",
    "second_half_goal_share",
    "months_active",
    "positions_played",
)


def _season_of(dates: pd.Series, start_month: int = 8) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    return (parsed.dt.year - (parsed.dt.month < start_month)).astype("Int64")


def match_level_features(appearances: pd.DataFrame, *, start_month: int = 8) -> pd.DataFrame:
    """Consistency, trajectory and rhythm, from the per-match rows."""
    frame = appearances.loc[:, ["player_id", "date", "minutes_played", "goals", "game_id"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["season"] = _season_of(frame["date"], start_month)
    frame = frame.dropna(subset=["season", "date"]).astype({"season": "int64"})

    # Calendar position within the player's own season, so "second half" means
    # after the winter break for a European league and something sensible for a
    # summer-calendar one, without hard-coding either.
    bounds = frame.groupby(["player_id", "season"])["date"].agg(["min", "max"])
    frame = frame.join(bounds, on=["player_id", "season"])
    span = (frame["max"] - frame["min"]).dt.days.clip(lower=1)
    frame["late"] = ((frame["date"] - frame["min"]).dt.days / span) >= 0.5

    grouped = frame.groupby(["player_id", "season"], as_index=False)
    out = grouped.agg(
        full_matches=("minutes_played", lambda s: int((s >= FULL_MATCH_MINUTES).sum())),
        minutes_variability=("minutes_played", "std"),
        scoring_matches=("goals", lambda s: int((s > 0).sum())),
        months_active=("date", lambda s: int(s.dt.to_period("M").nunique())),
        matches_counted=("game_id", "count"),
    )

    late_goals = frame[frame["late"]].groupby(["player_id", "season"])["goals"].sum()
    all_goals = frame.groupby(["player_id", "season"])["goals"].sum()
    share = (late_goals.reindex(all_goals.index).fillna(0) / all_goals.replace(0, np.nan)).rename(
        "second_half_goal_share"
    )
    out = out.merge(share.reset_index(), on=["player_id", "season"], how="left")

    out["full_match_share"] = out["full_matches"] / out["matches_counted"].clip(lower=1)
    out["scoring_match_share"] = out["scoring_matches"] / out["matches_counted"].clip(lower=1)
    logger.info("match-level features for %d player-seasons", len(out))
    return out.drop(columns=["full_matches", "scoring_matches"])


def squad_role(lineups: pd.DataFrame, *, start_month: int = 8) -> pd.DataFrame:
    """Starts, substitute appearances, captaincy and positional versatility.

    ``game_lineups.csv`` begins in 2013, so earlier seasons produce no rows and
    join to null downstream. That is the honest representation: the role is not
    zero, it is unknown.
    """
    frame = lineups.loc[:, ["date", "player_id", "type", "position", "team_captain"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["season"] = _season_of(frame["date"], start_month)
    frame = frame.dropna(subset=["season"]).astype({"season": "int64"})

    frame["is_start"] = frame["type"].eq("starting_lineup")
    frame["is_sub"] = frame["type"].eq("substitutes")

    out = frame.groupby(["player_id", "season"], as_index=False).agg(
        starts=("is_start", "sum"),
        substitute_appearances=("is_sub", "sum"),
        captain_matches=("team_captain", "sum"),
        positions_played=("position", "nunique"),
    )
    listed = out["starts"] + out["substitute_appearances"]
    out["start_share"] = out["starts"] / listed.clip(lower=1)
    out["captain_share"] = out["captain_matches"] / listed.clip(lower=1)
    logger.info("squad role for %d player-seasons (from %d lineup rows)", len(out), len(frame))
    return out.drop(columns=["captain_matches"])


def squad_match_counts(club_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """How many matches each club played per season — the availability denominator."""
    joined = club_games.loc[:, ["game_id", "club_id"]].merge(
        games.loc[:, ["game_id", "season"]], on="game_id", how="left"
    )
    joined = joined.dropna(subset=["season"]).astype({"season": "int64"})
    return joined.groupby(["club_id", "season"], as_index=False).agg(
        club_season_matches=("game_id", "nunique")
    )


def attach_performance(
    table: pd.DataFrame,
    *,
    match_features: pd.DataFrame,
    role: pd.DataFrame,
    squad_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Join the richer performance features and derive the ones that need both sides."""
    out = table.merge(match_features, on=["player_id", "season"], how="left")
    out = out.merge(role, on=["player_id", "season"], how="left")
    out = out.merge(
        squad_matches.rename(columns={"club_id": "primary_club_id"}),
        on=["primary_club_id", "season"],
        how="left",
    )

    out["goal_contributions"] = out["goals"] + out["assists"]
    per_90 = 90.0 / out["minutes_played"].clip(lower=90.0)
    out["contributions_per_90"] = out["goal_contributions"] * per_90
    # Capped at 1: a player can appear for two clubs in a season, so his match
    # count can exceed either club's total. Above 1 the ratio stops meaning
    # "how available was he" and starts meaning "he moved", which
    # `competitions_played` already carries.
    out["squad_match_share"] = (out["appearances"] / out["club_season_matches"].clip(lower=1)).clip(
        upper=1.0
    )
    return out.drop(columns=["matches_counted", "club_season_matches"], errors="ignore")


__all__ = [
    "FULL_MATCH_MINUTES",
    "PERFORMANCE_NUMERIC",
    "attach_performance",
    "match_level_features",
    "squad_match_counts",
    "squad_role",
]
