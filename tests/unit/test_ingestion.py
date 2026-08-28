"""Ingestion: the Kaggle source, the StatsBomb placeholder, and the pipeline.

No test here touches the network. The Kaggle source is driven with a fake HTTP
client that copies from data/sample/ and counts its calls, which is what makes
the "cached runs make no request" claim testable rather than asserted.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.base import DataSource, SourceUnavailableError
from src.ingestion.kaggle_loader import KaggleSource, resolve_credentials
from src.ingestion.statsbomb import StatsBombSource
from src.pipelines.ingest import ingest
from src.storage.duckdb_store import DuckDBParquetStore

FILES = ("players.csv", "player_valuations.csv", "appearances.csv")


class FakeHttpClient:
    """Stands in for HttpClient, serving data/sample/ and counting downloads."""

    def __init__(self, sample_dir: Path) -> None:
        self.sample_dir = sample_dir
        self.downloads: list[str] = []

    def download(self, url: str, destination: Path, **kwargs: object) -> Path:  # noqa: ARG002
        self.downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.sample_dir / Path(url).name, destination)
        return destination

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


@pytest.fixture
def source(tmp_path: Path, sample_dir: Path) -> tuple[KaggleSource, FakeHttpClient]:
    client = FakeHttpClient(sample_dir)
    return (
        KaggleSource(
            dataset="davidcariboo/player-scores",
            files=FILES,
            raw_dir=tmp_path / "raw",
            client=client,  # type: ignore[arg-type]
            credentials=("user", "key"),
        ),
        client,
    )


def test_kaggle_source_satisfies_the_protocol(
    source: tuple[KaggleSource, FakeHttpClient],
) -> None:
    assert isinstance(source[0], DataSource)


def test_available_tables_are_csv_stems(source: tuple[KaggleSource, FakeHttpClient]) -> None:
    assert source[0].available_tables() == ("players", "player_valuations", "appearances")


def test_fetch_downloads_every_file_once(source: tuple[KaggleSource, FakeHttpClient]) -> None:
    src, client = source
    paths = src.fetch()
    assert len(client.downloads) == 3
    assert all(p.is_file() for p in paths.values())


def test_second_fetch_makes_no_request(source: tuple[KaggleSource, FakeHttpClient]) -> None:
    """The cache requirement from the plan, asserted rather than assumed."""
    src, client = source
    src.fetch()
    assert len(client.downloads) == 3
    src.fetch()
    assert len(client.downloads) == 3, "a cached fetch hit the network"


def test_force_redownloads(source: tuple[KaggleSource, FakeHttpClient]) -> None:
    src, client = source
    src.fetch()
    src.fetch(force=True)
    assert len(client.downloads) == 6


def test_stale_cache_is_refetched(source: tuple[KaggleSource, FakeHttpClient]) -> None:
    src, client = source
    src.fetch()
    old = time.time() - (src.max_age_days + 1) * 86_400
    for filename in FILES:
        import os

        os.utime(src.raw_dir / filename, (old, old))
    src.fetch()
    assert len(client.downloads) == 6


def test_zero_byte_file_is_never_considered_fresh(
    source: tuple[KaggleSource, FakeHttpClient],
) -> None:
    """What an interrupted download used to leave behind."""
    src, _ = source
    path = src.raw_dir / "players.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert not src.is_fresh(path)


def test_load_returns_a_frame(source: tuple[KaggleSource, FakeHttpClient]) -> None:
    frame = source[0].load("players")
    assert isinstance(frame, pd.DataFrame)
    assert "player_id" in frame.columns
    assert len(frame) == 200


def test_path_for_unknown_table_raises(source: tuple[KaggleSource, FakeHttpClient]) -> None:
    with pytest.raises(KeyError):
        source[0].path_for("nope")


def test_missing_credentials_raise_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ingestion.kaggle_loader.KAGGLE_JSON", Path("/nonexistent/kaggle.json"))
    with pytest.raises(SourceUnavailableError, match="no Kaggle credentials"):
        resolve_credentials(env={})


def test_environment_credentials_win_over_the_token_file() -> None:
    """A deployment injecting credentials must outrank a stale local token."""
    assert resolve_credentials(env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}) == ("u", "k")


# --- StatsBomb placeholder ---


def test_statsbomb_satisfies_the_protocol() -> None:
    assert isinstance(StatsBombSource(), DataSource)


def test_statsbomb_offers_no_tables() -> None:
    assert StatsBombSource().available_tables() == ()


@pytest.mark.parametrize("call", [lambda s: s.fetch(), lambda s: s.load("anything")])
def test_statsbomb_raises_with_an_explanation(call: object) -> None:
    with pytest.raises(SourceUnavailableError, match="placeholder"):
        call(StatsBombSource())  # type: ignore[operator]


# --- pipeline ---


def test_ingest_writes_every_table_to_the_store(
    source: tuple[KaggleSource, FakeHttpClient], tmp_path: Path
) -> None:
    store = DuckDBParquetStore(tmp_path / "processed")
    report = ingest(source[0], store)

    assert store.list_tables() == ("appearances", "player_valuations", "players")
    assert report.tables["players"] == 200
    assert report.total_rows == sum(report.tables.values())


def test_ingested_tables_are_queryable_together(
    source: tuple[KaggleSource, FakeHttpClient], tmp_path: Path
) -> None:
    """The join the whole project depends on: valuations and appearances share
    players.player_id, so no entity resolution is needed."""
    store = DuckDBParquetStore(tmp_path / "processed")
    ingest(source[0], store)

    result = store.query(
        "SELECT count(DISTINCT p.player_id) AS n "
        "FROM players p "
        "JOIN player_valuations v ON v.player_id = p.player_id "
        "JOIN appearances a ON a.player_id = p.player_id"
    )
    assert result.iloc[0, 0] > 0


def test_ingest_report_renders(source: tuple[KaggleSource, FakeHttpClient], tmp_path: Path) -> None:
    report = ingest(source[0], DuckDBParquetStore(tmp_path / "p"))
    rendered = report.render()
    assert "players" in rendered
    assert "total" in rendered
