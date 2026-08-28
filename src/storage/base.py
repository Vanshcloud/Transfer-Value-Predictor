"""The storage contract every consumer talks to.

Nothing outside ``src/storage`` imports DuckDB. That is the whole point: the
project runs on Parquet today, and adding PostgreSQL later means writing a
second implementation of this Protocol rather than editing the pipeline.

One Protocol, one implementation. There is no registry, no factory and no
plugin system, because there is exactly one backend and speculative
indirection is harder to delete than to add.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TableStore(Protocol):
    """A named-table store backed by Parquet, a database, or anything else."""

    def write_table(self, name: str, frame: pd.DataFrame) -> None:
        """Persist ``frame`` under ``name``, replacing any existing table."""
        ...

    def read_table(self, name: str) -> pd.DataFrame:
        """Return the table stored under ``name``.

        Raises:
            KeyError: if no such table exists.
        """
        ...

    def has_table(self, name: str) -> bool:
        """Whether ``name`` exists in the store."""
        ...

    def list_tables(self) -> tuple[str, ...]:
        """Every table name in the store, sorted."""
        ...

    def query(self, sql: str) -> pd.DataFrame:
        """Run a read-only SQL query and return the result.

        Table names in ``sql`` refer to tables in this store.
        """
        ...
