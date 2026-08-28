"""The Parquet/DuckDB store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.storage.base import TableStore
from src.storage.duckdb_store import DuckDBParquetStore


@pytest.fixture
def store(tmp_path: Path) -> DuckDBParquetStore:
    return DuckDBParquetStore(tmp_path / "processed")


def test_implementation_satisfies_the_protocol(store: DuckDBParquetStore) -> None:
    assert isinstance(store, TableStore)


def test_write_then_read_round_trips(store: DuckDBParquetStore) -> None:
    frame = pd.DataFrame({"player_id": [1, 2], "value": [10.5, 20.5]})
    store.write_table("players", frame)
    pd.testing.assert_frame_equal(store.read_table("players"), frame)


def test_write_replaces_an_existing_table(store: DuckDBParquetStore) -> None:
    store.write_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    store.write_table("t", pd.DataFrame({"a": [9]}))
    assert len(store.read_table("t")) == 1


def test_reading_a_missing_table_raises_keyerror(store: DuckDBParquetStore) -> None:
    with pytest.raises(KeyError):
        store.read_table("absent")


def test_has_table_and_list_tables(store: DuckDBParquetStore) -> None:
    assert store.list_tables() == ()
    assert not store.has_table("players")
    store.write_table("players", pd.DataFrame({"a": [1]}))
    store.write_table("clubs", pd.DataFrame({"a": [1]}))
    assert store.has_table("players")
    assert store.list_tables() == ("clubs", "players")


def test_query_joins_across_tables(store: DuckDBParquetStore) -> None:
    store.write_table("players", pd.DataFrame({"player_id": [1, 2], "name": ["a", "b"]}))
    store.write_table("vals", pd.DataFrame({"player_id": [1, 1, 2], "v": [10, 20, 30]}))
    result = store.query(
        "SELECT p.name, count(*) AS n FROM players p JOIN vals v USING (player_id) "
        "GROUP BY p.name ORDER BY p.name"
    )
    assert result.to_dict("records") == [{"name": "a", "n": 2}, {"name": "b", "n": 1}]


def test_query_works_when_the_path_contains_a_quote(tmp_path: Path) -> None:
    """The store path comes from DATA_DIR, and a directory may contain a quote."""
    store = DuckDBParquetStore(tmp_path / "o'brien")
    store.write_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    assert store.query("SELECT count(*) AS n FROM t").iloc[0, 0] == 3


@pytest.mark.parametrize("name", ["../escape", "a; DROP TABLE x", "a'b", "1abc", "", "a b"])
def test_unsafe_table_names_are_rejected(store: DuckDBParquetStore, name: str) -> None:
    """Names become filenames and are interpolated into SQL, so they are
    restricted rather than escaped."""
    with pytest.raises(ValueError):
        store.path_for(name)


def test_partial_files_are_not_listed_as_tables(store: DuckDBParquetStore) -> None:
    """An interrupted write leaves a .parquet.part, which must not read as a table."""
    store.write_table("good", pd.DataFrame({"a": [1]}))
    (store.root / "bad.parquet.part").write_bytes(b"not parquet")
    assert store.list_tables() == ("good",)
