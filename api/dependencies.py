"""Application state and how handlers reach it.

The service is built once in the lifespan handler and injected with ``Depends``.
Not at import time: an import-time load makes this module unimportable without
a trained model, which would break collection of every test in the suite —
including the ones that have nothing to do with the API.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import Depends, Request

from src.pipelines.features import CURRENT_SEASON_TABLE, TRAINING_TABLE
from src.services.prediction import Club, PredictionService
from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def club_index(players: pd.DataFrame, competitions: pd.DataFrame | None) -> dict[int, Club]:
    """Club, league and league country for every player, keyed by player id.

    Joined here rather than in the feature build, for the same reason names
    are: where a man plays is a label on a search result, not a feature the
    model is allowed to see.

    A missing or column-less `competitions` table costs the league and leaves
    the club, and a competition id with no row (the dataset ships a few) costs
    the league for that player alone. Neither is worth refusing to serve over.
    """
    if "current_club_name" not in players.columns:
        return {}

    leagues: dict[str, tuple[str, str | None]] = {}
    if competitions is not None and {"competition_id", "name"} <= set(competitions.columns):
        for identifier, name, country in zip(
            competitions["competition_id"],
            competitions["name"],
            competitions.get("country_name", pd.Series(dtype=object)).reindex(competitions.index),
            strict=True,
        ):
            if pd.isna(identifier) or pd.isna(name):
                continue
            # The dataset spells a competition as its URL slug — "premier-league",
            # "j1-league". Title-cased rather than mapped to a hand-written
            # display name, because sponsors rename these leagues between
            # dataset refreshes and a hardcoded table would go quietly stale.
            leagues[str(identifier)] = (
                str(name).replace("-", " ").title(),
                None if pd.isna(country) else str(country),
            )

    index: dict[int, Club] = {}
    competition_ids = players.get(
        "current_club_domestic_competition_id", pd.Series(dtype=object)
    ).reindex(players.index)
    for player_id, club, competition in zip(
        players["player_id"], players["current_club_name"], competition_ids, strict=True
    ):
        league, country = leagues.get(str(competition), (None, None))
        name = None if pd.isna(club) else str(club)
        if name is None and league is None:
            continue
        index[int(player_id)] = Club(name, league, country)
    return index


def build_service(settings: Settings) -> PredictionService:
    """Load the models, and the player history if it happens to be there.

    A missing training table is not fatal. The service still predicts from
    explicit features and still reports model metadata; only player lookup is
    unavailable. Refusing to start would turn a partial deployment into no
    deployment.
    """
    players: pd.DataFrame | None = None
    current: pd.DataFrame | None = None
    names: dict[int, str] = {}
    clubs: dict[int, Club] = {}
    try:
        store = DuckDBParquetStore(settings.paths.processed_dir)
        if store.has_table(TRAINING_TABLE):
            players = store.read_table(TRAINING_TABLE)
            logger.info("loaded %d player-season rows", len(players))
        else:
            logger.warning("no %s table; player lookup will be unavailable", TRAINING_TABLE)

        # The season being played: features complete, label not yet published.
        # Optional — a deployment built before this table existed still serves.
        if store.has_table(CURRENT_SEASON_TABLE):
            current = store.read_table(CURRENT_SEASON_TABLE)
            logger.info("loaded %d current-season rows", len(current))
        else:
            logger.info(
                "no %s table; predictions stop at the last labelled season", CURRENT_SEASON_TABLE
            )

        # Names live in the raw players table, not the training table — the
        # feature build drops them deliberately, since a name is not a feature.
        # Search needs them, so they are joined back here rather than pushed
        # into the model's input.
        if store.has_table("players"):
            raw = store.read_table("players")
            names = dict(zip(raw["player_id"], raw["name"], strict=True))
            logger.info("loaded %d player names", len(names))
            # The club and its league come from the same table for the same
            # reason. Names repeat in this dataset and a list of them alone
            # gives a reader no way to tell which one he was looking for.
            competitions = (
                store.read_table("competitions") if store.has_table("competitions") else None
            )
            clubs = club_index(raw, competitions)
            logger.info("loaded clubs for %d players", len(clubs))
        else:
            logger.warning("no players table; name search will be unavailable")
    except OSError as exc:  # pragma: no cover - depends on the deployment
        logger.warning("could not read player data: %s", exc)

    return PredictionService.from_directory(
        settings.paths.model_dir, players, names, current_season=current, clubs=clubs
    )


def get_service(request: Request) -> PredictionService:
    """The process-wide service, put on app state by the lifespan handler."""
    service: PredictionService = request.app.state.service
    return service


ServiceDep = Annotated[PredictionService, Depends(get_service)]
"""What every handler asks for. One alias so the injection point is stated once."""
