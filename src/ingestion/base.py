"""The contract every data provider satisfies.

The pipeline downstream of ingestion is source-agnostic: it asks a source for a
named table and receives a DataFrame. Adding a provider means implementing this
Protocol, not editing anything that consumes it.

A source is responsible for its own caching and for its own politeness. It is
not responsible for cleaning or feature engineering — those are later stages,
and a source that quietly reshapes its output makes the pipeline's behaviour
depend on which provider supplied the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    """A provider of football data."""

    @property
    def name(self) -> str:
        """Short identifier used in logs and in the raw-data directory layout."""
        ...

    def available_tables(self) -> tuple[str, ...]:
        """Table names this source can supply."""
        ...

    def fetch(self, *, force: bool = False) -> dict[str, Path]:
        """Ensure the raw files are on disk and return name -> path.

        Implementations skip work when a cached copy is fresh, unless ``force``.
        """
        ...

    def load(self, table: str) -> pd.DataFrame:
        """Read one table into memory, fetching it first if necessary."""
        ...


class SourceUnavailableError(RuntimeError):
    """A source exists in the codebase but cannot currently supply data.

    Distinct from a network failure: this means the provider is a placeholder,
    or is gated behind credentials or terms the project does not satisfy.
    """
