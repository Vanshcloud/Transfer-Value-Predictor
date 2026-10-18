"""``python -m api``: serve on the configured address.

The only reader of ``api.host`` and ``api.port`` (and so of ``API_HOST`` and
``API_PORT``). Before Phase 20 both were documented and read by nothing: the
image hard-coded its address in a uvicorn command line.

    python -m api            # the Docker image
    python -m api --reload   # development (`make serve`)
"""

from __future__ import annotations

import argparse

import uvicorn

from src.utils.config import load_settings

LIMIT_CONCURRENCY = 64
"""Open connections before uvicorn answers 503 rather than queueing without
bound. Model work itself is capped lower, per core, in ``api/limits.py``."""

TIMEOUT_KEEP_ALIVE = 5
"""Seconds an idle keep-alive connection is held open."""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args(argv)

    settings = load_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=args.reload,
        limit_concurrency=LIMIT_CONCURRENCY,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )


if __name__ == "__main__":
    main()
