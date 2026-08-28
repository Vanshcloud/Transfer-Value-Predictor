"""Parquet-on-disk storage, queried through DuckDB.

Chosen over PostgreSQL for the early phases because the entire dataset is a few
hundred megabytes of columnar files: DuckDB reads Parquet directly with no
server, no schema migrations and no container to keep running. When the API
needs indexed player lookup, a ``PostgresStore`` implements the same
:class:`~src.storage.base.TableStore` Protocol and callers do not change.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Table names become filenames and are interpolated into SQL, so they are
# restricted rather than escaped. A name that cannot express a path traversal
# or a quote cannot be used for one.
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_name(name: str) -> str:
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"invalid table name {name!r}: expected letters, digits and underscores, "
            "starting with a letter or underscore"
        )
    return name


class DuckDBParquetStore:
    """A :class:`~src.storage.base.TableStore` over a directory of Parquet files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        """The Parquet file backing ``name``."""
        return self.root / f"{_validate_name(name)}.parquet"

    def write_table(self, name: str, frame: pd.DataFrame) -> None:
        path = self.path_for(name)
        # Write beside the target and rename, so a crash mid-write cannot leave
        # a half-written file that later reads as a valid-looking table.
        partial = path.with_suffix(".parquet.part")
        frame.to_parquet(partial, index=False)
        partial.replace(path)
        logger.info("wrote table %s (%d rows, %d cols)", name, len(frame), frame.shape[1])

    def read_table(self, name: str) -> pd.DataFrame:
        path = self.path_for(name)
        if not path.exists():
            raise KeyError(f"no such table: {name}")
        return pd.read_parquet(path)

    def has_table(self, name: str) -> bool:
        return self.path_for(name).exists()

    def list_tables(self) -> tuple[str, ...]:
        return tuple(sorted(p.stem for p in self.root.glob("*.parquet")))

    def query(self, sql: str) -> pd.DataFrame:
        """Run ``sql`` with every stored table registered as a view."""
        with duckdb.connect(database=":memory:") as connection:
            for name in self.list_tables():
                # DuckDB rejects a prepared parameter in CREATE VIEW, so the
                # path is inlined. It is a SQL string literal, so single quotes
                # are doubled — the project root is user-supplied via DATA_DIR
                # and a directory may legally contain one. The table name needs
                # no escaping: _validate_name already restricts it.
                literal = str(self.path_for(name)).replace("'", "''")
                connection.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{literal}')")
            return connection.execute(sql).fetch_df()
