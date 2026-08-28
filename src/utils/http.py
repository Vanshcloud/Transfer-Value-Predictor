"""A polite, retrying HTTP client shared by every ingestion module.

One client for all outbound traffic, so timeout, retry, backoff, User-Agent and
rate-limit policy are set in one place rather than rediscovered per source.

`requests` only. Playwright is a last resort reserved for a source that leaves
no alternative, and Selenium is not used at all.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Transient failures only. A 404 is not retried: the file is missing, and
# asking four more times just delays the error by the backoff schedule.
RETRY_STATUSES = (429, 500, 502, 503, 504)


class HttpClient:
    """A `requests.Session` with retry, backoff, timeout and rate limiting.

    Usable as a context manager so the underlying connection pool is closed.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        user_agent: str = "transfer-value-predictor/0.1",
        min_request_interval_seconds: float = 0.0,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_request_at: float | None = None

        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=RETRY_STATUSES,
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _wait_for_rate_limit(self) -> None:
        if self.min_request_interval_seconds <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_request_interval_seconds - elapsed
        if remaining > 0:
            logger.debug("rate limit: sleeping %.2fs", remaining)
            time.sleep(remaining)

    def get(self, url: str, **kwargs: object) -> requests.Response:
        """GET ``url``, honouring the configured timeout and rate limit."""
        self._wait_for_rate_limit()
        kwargs.setdefault("timeout", self.timeout_seconds)
        try:
            response = self.session.get(url, **kwargs)  # type: ignore[arg-type]
        finally:
            self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def download(self, url: str, destination: Path, **kwargs: object) -> Path:
        """Stream ``url`` to ``destination``.

        Written to a ``.part`` file and renamed only on success, so an
        interrupted download can never be mistaken for a complete one by the
        cache-freshness check.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")

        kwargs.setdefault("stream", True)
        logger.info("downloading %s -> %s", url, destination.name)
        response = self.get(url, **kwargs)

        # Any failure below must remove the .part file. A short read that stays
        # on disk is worse than no file at all: the cache-freshness check only
        # asks whether a file exists and is non-empty, so a partial download
        # would be served as though it were complete. urllib3 raises
        # ChunkedEncodingError mid-stream when a response undershoots its
        # declared Content-Length, so the explicit size check below is a
        # backstop for the case where it does not.
        try:
            bytes_written = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        bytes_written += len(chunk)

            expected = response.headers.get("Content-Length")
            if expected is not None and bytes_written != int(expected):
                raise OSError(
                    f"truncated download for {destination.name}: "
                    f"got {bytes_written} bytes, expected {expected}"
                )
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

        partial.replace(destination)
        logger.info("downloaded %s (%.1f MB)", destination.name, bytes_written / 1e6)
        return destination

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
