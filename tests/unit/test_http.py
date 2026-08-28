"""The shared HTTP client.

Served by a local http.server on a loopback port rather than by mocking
requests: retry, streaming and Content-Length handling are behaviours of the
real stack, and a mock that returns whatever the test wants proves nothing
about them.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests

from src.utils.http import HttpClient

PAYLOAD = b"player_id,name\n1,someone\n" * 100


class _Handler(BaseHTTPRequestHandler):
    """Serves /ok, /flaky (503 twice then 200), /missing, and /truncated."""

    attempts: dict[str, int] = {}

    def log_message(self, *args: object) -> None:  # silence the test output
        pass

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        count = self.attempts.get(self.path, 0) + 1
        self.attempts[self.path] = count

        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
        elif self.path == "/flaky":
            if count < 3:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(PAYLOAD)
        elif self.path == "/truncated":
            # Declares more than it sends — the case that used to produce a
            # short file that later looked like a valid cached download.
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD) * 2))
            self.end_headers()
            self.wfile.write(PAYLOAD)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _reset_attempts() -> None:
    _Handler.attempts.clear()


def test_get_returns_the_body(server: str) -> None:
    with HttpClient() as client:
        assert client.get(f"{server}/ok").content == PAYLOAD


def test_user_agent_is_sent() -> None:
    with HttpClient(user_agent="tvp-test/1.0") as client:
        assert client.session.headers["User-Agent"] == "tvp-test/1.0"


def test_transient_5xx_is_retried_until_it_succeeds(server: str) -> None:
    with HttpClient(max_retries=5, backoff_factor=0) as client:
        assert client.get(f"{server}/flaky").content == PAYLOAD
    assert _Handler.attempts["/flaky"] == 3


def test_404_is_not_retried(server: str) -> None:
    """Retrying a missing file only delays the error."""
    with (
        HttpClient(max_retries=5, backoff_factor=0) as client,
        pytest.raises(requests.HTTPError),
    ):
        client.get(f"{server}/missing")
    assert _Handler.attempts["/missing"] == 1


def test_download_writes_the_file(server: str, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.csv"
    with HttpClient() as client:
        client.download(f"{server}/ok", target)
    assert target.read_bytes() == PAYLOAD


def test_truncated_download_raises_and_leaves_nothing_behind(server: str, tmp_path: Path) -> None:
    """The cache-freshness check trusts any non-empty file, so a short read must
    never be allowed to land at the final path."""
    target = tmp_path / "out.csv"
    # urllib3 detects the undershoot mid-stream and raises ChunkedEncodingError
    # before the explicit size check is reached; the check remains as a
    # backstop. Either way nothing may be left on disk.
    with HttpClient() as client, pytest.raises((OSError, requests.RequestException)):
        client.download(f"{server}/truncated", target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "a partial download was left on disk"


def test_rate_limit_delays_the_second_request(server: str) -> None:
    import time

    with HttpClient(min_request_interval_seconds=0.25) as client:
        client.get(f"{server}/ok")
        started = time.monotonic()
        client.get(f"{server}/ok")
        assert time.monotonic() - started >= 0.2


def test_no_rate_limit_by_default(server: str) -> None:
    import time

    with HttpClient() as client:
        client.get(f"{server}/ok")
        started = time.monotonic()
        client.get(f"{server}/ok")
        assert time.monotonic() - started < 0.2
