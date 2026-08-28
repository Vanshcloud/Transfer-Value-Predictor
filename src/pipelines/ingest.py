"""Raw CSV in, Parquet tables out.

The only stage that touches the network. Everything downstream reads from the
:class:`~src.storage.base.TableStore`, so a rerun of the pipeline needs no
credentials once the raw files are cached.

Deliberately does no cleaning. Validation is the next stage and needs to see the
data exactly as the provider published it — a converter that quietly coerces
dtypes hides precisely the problems validation exists to find.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ingestion.base import DataSource
from src.storage.base import TableStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestReport:
    """What one ingestion run produced."""

    tables: dict[str, int]
    """Table name -> row count."""

    @property
    def total_rows(self) -> int:
        return sum(self.tables.values())

    def render(self) -> str:
        lines = [f"  {name:<22} {rows:>10,} rows" for name, rows in sorted(self.tables.items())]
        return "\n".join([*lines, f"  {'total':<22} {self.total_rows:>10,} rows"])


def ingest(
    source: DataSource,
    store: TableStore,
    *,
    force: bool = False,
) -> IngestReport:
    """Fetch every table ``source`` offers and write it to ``store``."""
    source.fetch(force=force)

    counts: dict[str, int] = {}
    for table in source.available_tables():
        frame: pd.DataFrame = source.load(table)
        store.write_table(table, frame)
        counts[table] = len(frame)
        logger.info("ingested %s: %d rows, %d columns", table, len(frame), frame.shape[1])

    return IngestReport(tables=counts)
