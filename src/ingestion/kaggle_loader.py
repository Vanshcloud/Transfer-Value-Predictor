"""Kaggle ``davidcariboo/player-scores`` — the project's primary source.

This is a CC0 Public Domain mirror of Transfermarkt data, refreshed weekly. It
is used *instead of* scraping Transfermarkt directly, whose Terms of Use §11.1
prohibit automated access and separately prohibit using the content to train
machine-learning models (plans/00-discovery.md §1.1).

Downloads go through the Kaggle REST API with ``requests`` rather than the
``kaggle`` package: the endpoint is a single authenticated GET per file, so the
dependency would buy nothing but its own credential-discovery quirks. The API
redirects to a signed Google Storage URL; ``requests`` drops the Authorization
header on a cross-host redirect, which is correct here because the signed URL
carries its own credentials.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from src.ingestion.base import SourceUnavailableError
from src.utils.http import HttpClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

KAGGLE_API_ROOT = "https://www.kaggle.com/api/v1"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def resolve_credentials(env: dict[str, str] | None = None) -> tuple[str, str]:
    """Find Kaggle credentials, preferring the environment over the token file.

    Order matters: a deployment injecting ``KAGGLE_USERNAME``/``KAGGLE_KEY``
    should win over a developer's stale ``~/.kaggle/kaggle.json``.

    Raises:
        SourceUnavailableError: if no usable credentials are found.
    """
    source = os.environ if env is None else env
    username = source.get("KAGGLE_USERNAME")
    key = source.get("KAGGLE_KEY")
    if username and key:
        return username, key

    if KAGGLE_JSON.is_file():
        try:
            payload = json.loads(KAGGLE_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceUnavailableError(f"{KAGGLE_JSON} is unreadable: {exc}") from exc
        username, key = payload.get("username"), payload.get("key")
        if username and key:
            return username, key

    raise SourceUnavailableError(
        "no Kaggle credentials: set KAGGLE_USERNAME and KAGGLE_KEY (see .env.example), "
        f"or place a token at {KAGGLE_JSON}"
    )


def _age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86_400


class KaggleSource:
    """Fetches CSVs from one Kaggle dataset and exposes them as named tables.

    Table names are the CSV stems: ``players.csv`` becomes ``players``.
    """

    def __init__(
        self,
        dataset: str,
        files: tuple[str, ...],
        raw_dir: Path,
        *,
        max_age_days: int = 7,
        client: HttpClient | None = None,
        credentials: tuple[str, str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.files = tuple(files)
        self.raw_dir = Path(raw_dir) / "kaggle"
        self.max_age_days = max_age_days
        self._client = client
        self._credentials = credentials

    @property
    def name(self) -> str:
        return "kaggle"

    def available_tables(self) -> tuple[str, ...]:
        return tuple(Path(f).stem for f in self.files)

    def path_for(self, table: str) -> Path:
        """Local path of the CSV backing ``table``."""
        for filename in self.files:
            if Path(filename).stem == table:
                return self.raw_dir / filename
        raise KeyError(f"unknown table {table!r}; expected one of {self.available_tables()}")

    def is_fresh(self, path: Path) -> bool:
        """Whether a cached file exists and is newer than ``max_age_days``.

        A zero-byte file is never fresh — that is what an interrupted download
        used to leave behind before downloads became atomic.
        """
        if not path.is_file() or path.stat().st_size == 0:
            return False
        return _age_days(path) < self.max_age_days

    def fetch(self, *, force: bool = False) -> dict[str, Path]:
        """Download any file that is missing or stale. Returns table -> path."""
        paths: dict[str, Path] = {}
        stale = [f for f in self.files if force or not self.is_fresh(self.raw_dir / f)]

        if not stale:
            logger.info(
                "kaggle: all %d files cached and fresh, nothing to download", len(self.files)
            )
            return {Path(f).stem: self.raw_dir / f for f in self.files}

        credentials = self._credentials or resolve_credentials()
        client = self._client or HttpClient()
        try:
            for filename in self.files:
                destination = self.raw_dir / filename
                table = Path(filename).stem
                if filename not in stale:
                    logger.debug("kaggle: %s is fresh, skipping", filename)
                    paths[table] = destination
                    continue
                url = f"{KAGGLE_API_ROOT}/datasets/download/{self.dataset}/{filename}"
                paths[table] = client.download(url, destination, auth=credentials)
        finally:
            if self._client is None:
                client.close()
        return paths

    def load(self, table: str) -> pd.DataFrame:
        """Read ``table``, downloading it first if it is missing or stale."""
        path = self.path_for(table)
        if not self.is_fresh(path):
            self.fetch()
        return pd.read_csv(path)
