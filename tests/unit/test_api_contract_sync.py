"""The contract document and the running app must list the same endpoints.

Phase 13 found two routes — `/players/{player_id}/history` and
`/features/distribution` — live in the app and absent from
`docs/API_CONTRACT.md`. They had been shipped, tested and served for a week
without the document knowing. A doc that is only checked by reading it is
checked once, on the day it is written.
"""

from __future__ import annotations

import re
from pathlib import Path

from api.main import create_app

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "API_CONTRACT.md"

# The endpoint table's rows: | `GET` | `/api/v1/players?q=` | ... |
# Trailing query strings are documentation, not part of the path.
ROW = re.compile(r"\| `(?:GET|POST|PUT|DELETE|PATCH)` \| `([^`?]+)")


def documented_paths() -> set[str]:
    return set(ROW.findall(CONTRACT.read_text()))


def test_every_live_route_is_documented() -> None:
    live = set(create_app().openapi()["paths"])
    assert not live - documented_paths()


def test_every_documented_route_exists() -> None:
    live = set(create_app().openapi()["paths"])
    assert not documented_paths() - live
