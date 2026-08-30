"""The contract, the README and the running app must list the same endpoints.

Phase 13 found two routes — `/players/{player_id}/history` and
`/features/distribution` — live in the app and absent from
`docs/API_CONTRACT.md`. They had been shipped, tested and served for a week
without the document knowing.

Phase 14's audit then found the *same two* missing from the README's endpoint
table, because fixing the contract had taught nothing about the other document
describing the same surface. So both are checked here, in both directions.
A doc that is only checked by reading it is checked once, on the day it is
written.
"""

from __future__ import annotations

import re
from pathlib import Path

from api.main import create_app

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "API_CONTRACT.md"
README = ROOT / "README.md"

# The endpoint table's rows: | `GET` | `/api/v1/players?q=` | ... |
# Trailing query strings are documentation, not part of the path.
ROW = re.compile(r"\| `(?:GET|POST|PUT|DELETE|PATCH)` \| `([^`?]+)")


def live_paths() -> set[str]:
    return set(create_app().openapi()["paths"])


def documented_paths(document: Path) -> set[str]:
    return set(ROW.findall(document.read_text()))


def test_every_live_route_is_in_the_contract() -> None:
    assert not live_paths() - documented_paths(CONTRACT)


def test_every_contract_route_exists() -> None:
    assert not documented_paths(CONTRACT) - live_paths()


def test_every_live_route_is_in_the_readme() -> None:
    """The README is the only document most readers will open."""
    assert not live_paths() - documented_paths(README)


def test_every_readme_route_exists() -> None:
    assert not documented_paths(README) - live_paths()


def test_the_readme_and_the_contract_agree() -> None:
    assert documented_paths(README) == documented_paths(CONTRACT)


NUMBER_WORDS = {
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
}


def test_the_prose_count_matches_the_table() -> None:
    """A number written out in prose drifts from the table under it.

    The README and the CHANGELOG both say how many endpoints there are. Adding
    a twelfth would leave two sentences quietly wrong, which is the same defect
    this file exists for — just in prose rather than in a table.
    """
    count = len(live_paths())
    word = NUMBER_WORDS[count]
    for document in (README, ROOT / "CHANGELOG.md"):
        body = document.read_text().lower()
        assert word in body, f"{document.name} should say '{word}' endpoints; it lists {count}"
        for other, wrong in NUMBER_WORDS.items():
            if other != count:
                assert (
                    f"{wrong} endpoint" not in body
                ), f"{document.name} says '{wrong} endpoints' but there are {count}"
