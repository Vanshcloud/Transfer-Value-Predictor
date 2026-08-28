"""StatsBomb open data — placeholder.

Deliberately not wired into the pipeline. The reasoning is recorded in
plans/00-discovery.md §1.4 and is worth keeping next to the code, because the
obvious assumption is that free event data must be useful here:

- ``statsbombpy.player_season_stats`` raises on open data ("there is currently
  no open data for aggregated stats"). The one model-ready function is
  credentialed, so using it would mean writing an event-to-season aggregation
  pipeline — the most expensive code in the project, for the fewest rows.
- Open data covers 80 competition-seasons skewed across five decades. The
  Premier League has two of them, 2003/04 and 2015/16. That does not overlap
  the market-value panel, which is 2011-2024 across many players.
- It introduces a third player-ID namespace to reconcile.

The module exists so that adding StatsBomb later is an implementation of an
existing Protocol rather than a refactor of the ingestion layer. It is a stub
on purpose: building an aggregation nothing consumes would be speculative work
that still has to be maintained.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.base import SourceUnavailableError

_REASON = (
    "StatsBomb is a placeholder source. Open data does not cover the seasons this "
    "project models, and its aggregated-stat API requires credentials. "
    "See plans/00-discovery.md section 1.4."
)


class StatsBombSource:
    """Conforms to :class:`~src.ingestion.base.DataSource`; supplies nothing yet."""

    @property
    def name(self) -> str:
        return "statsbomb"

    def available_tables(self) -> tuple[str, ...]:
        return ()

    # `force` and `table` are unused because both methods raise; they exist to
    # match the DataSource signature, which is the point of a placeholder.
    def fetch(self, *, force: bool = False) -> dict[str, Path]:  # noqa: ARG002
        raise SourceUnavailableError(_REASON)

    def load(self, table: str) -> pd.DataFrame:  # noqa: ARG002
        raise SourceUnavailableError(_REASON)
