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

from src.pipelines.features import TRAINING_TABLE
from src.services.prediction import PredictionService
from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_service(settings: Settings) -> PredictionService:
    """Load the models, and the player history if it happens to be there.

    A missing training table is not fatal. The service still predicts from
    explicit features and still reports model metadata; only player lookup is
    unavailable. Refusing to start would turn a partial deployment into no
    deployment.
    """
    players: pd.DataFrame | None = None
    try:
        store = DuckDBParquetStore(settings.paths.processed_dir)
        if store.has_table(TRAINING_TABLE):
            players = store.read_table(TRAINING_TABLE)
            logger.info("loaded %d player-season rows", len(players))
        else:
            logger.warning("no %s table; player lookup will be unavailable", TRAINING_TABLE)
    except OSError as exc:  # pragma: no cover - depends on the deployment
        logger.warning("could not read player data: %s", exc)

    return PredictionService.from_directory(settings.paths.model_dir, players)


def get_service(request: Request) -> PredictionService:
    """The process-wide service, put on app state by the lifespan handler."""
    service: PredictionService = request.app.state.service
    return service


ServiceDep = Annotated[PredictionService, Depends(get_service)]
"""What every handler asks for. One alias so the injection point is stated once."""
