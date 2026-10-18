"""Row counts the documentation states, against the tables actually saved.

The release re-audit found the methodology still quoting the prior-value count
from before the prior was redefined, and the dashboard's home page a count two
table rebuilds old. Read from ``data/processed/`` as saved — the tables the
shipped models were trained and frozen on — not rebuilt.
"""

from __future__ import annotations

import pytest

from src.storage.duckdb_store import DuckDBParquetStore
from src.utils.paths import PROJECT_ROOT

pytestmark = pytest.mark.integration

DOCS = PROJECT_ROOT / "docs"


@pytest.fixture(scope="module")
def store() -> DuckDBParquetStore:
    store = DuckDBParquetStore(PROJECT_ROOT / "data" / "processed")
    if not (store.has_table("training_table") and store.has_table("current_season")):
        pytest.skip("run scripts/build_features.py first")
    return store


def test_historical_counts(store: DuckDBParquetStore) -> None:
    table = store.read_table("training_table")
    rows, players = len(table), table["player_id"].nunique()
    prior = int(table["prev_market_value_in_eur"].notna().sum())
    card = (DOCS / "DATASET_CARD.md").read_text()
    assert f"| Rows | {rows:,} |" in card
    assert f"| Players | {players:,} |" in card
    assert f"{prior:,} ({prior / rows:.1%})" in card
    assert f"{prior:,} rows, {prior / rows:.1%}" in (DOCS / "METHODOLOGY.md").read_text()
    assert f"- Rows: {rows:,}" in (DOCS / "MODEL_CARD_performance_only.md").read_text()
    assert f"- Rows: {prior:,}" in (DOCS / "MODEL_CARD_with_prior_value.md").read_text()
    home = (PROJECT_ROOT / "frontend" / "src" / "app" / "page.tsx").read_text()
    assert f"over {rows:,} player-seasons" in home


def test_current_season_counts(store: DuckDBParquetStore) -> None:
    current = store.read_table("current_season")
    by_season = current.groupby("season").size()
    newest = int(by_season.index.max())
    pending = len(current) - int(by_season[newest])
    card = (DOCS / "DATASET_CARD.md").read_text()
    assert (
        f"{len(current):,}: {by_season[newest]:,} for season {newest}, "
        f"and {pending:,} season-{newest - 1} rows" in card
    )
    assert (
        f"{pending:,} season-{newest - 1} rows are still awaiting"
        in (PROJECT_ROOT / "README.md").read_text()
    )


def test_the_label_horizon_is_stated_as_measured(store: DuckDBParquetStore) -> None:
    horizon = store.read_table("training_table")["label_horizon_days"]
    q1, median, q3 = (round(horizon.quantile(q)) for q in (0.25, 0.5, 0.75))
    card = (DOCS / "DATASET_CARD.md").read_text()
    assert f"median {median} days, interquartile {q1}–{q3}" in card
    assert f"median {median} days" in (PROJECT_ROOT / "README.md").read_text()
    assert (
        f"median {median} days, interquartile range\n{q1}–{q3}"
        in (DOCS / "METHODOLOGY.md").read_text()
    )
