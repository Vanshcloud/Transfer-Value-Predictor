"""The limits a public, unauthenticated, read-only API enforces on itself.

Only what one process can enforce correctly lives here. Rate limiting and
request timeouts do not: a per-process counter is wrong as soon as there are
two workers or a proxy in front, and a Python thread running SHAP cannot be
interrupted. Those are the reverse proxy's job, listed in SECURITY.md.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.errors import ServiceBusyError, error_response

MAX_BODY_BYTES = 64 * 1024
"""The largest request body accepted. A prediction request with every feature
is under 4 KB; 64 KB leaves room for whitespace and long category names and
none for the 1 GB body Starlette would otherwise read into memory."""

MAX_CONCURRENT_INFERENCE = max(1, min(os.cpu_count() or 1, 8))
"""Model work in flight at once: one per core, up to eight. SHAP is CPU-bound,
so more threads than cores only makes every request slower."""

INFERENCE_WAIT_SECONDS = 2.0
"""How long a request waits for a slot before it is turned away. Long enough
to absorb a burst, short enough that a queue never builds behind a slow one."""


class BodySizeLimit:
    """Refuse a body over :data:`MAX_BODY_BYTES` with a 413, before the app runs.

    The declared ``Content-Length`` is checked first; a chunked body declares
    nothing, so it is read here up to the limit and replayed to the app only
    if it fits. Either way the app never sees an oversized body.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse(scope, receive, send)
            return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # A disconnect: pass it on and let the app end the request.
                replay = [message]
                break
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await self._refuse(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                replay = [{"type": "http.request", "body": b"".join(chunks), "more_body": False}]
                break

        async def replaying() -> Message:
            return replay.pop(0) if replay else await receive()

        await self.app(scope, replaying, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = error_response(
            "payload_too_large", f"request body exceeds {self.max_bytes} bytes", 413
        )
        await response(scope, receive, send)


_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)


def inference_slot() -> Iterator[None]:
    """A FastAPI dependency held for the duration of model work.

    Raises:
        ServiceBusyError: when every slot stays taken for
            :data:`INFERENCE_WAIT_SECONDS`. Mapped to 503 with ``Retry-After``.
    """
    if not _slots.acquire(timeout=INFERENCE_WAIT_SECONDS):
        raise ServiceBusyError(
            f"all {MAX_CONCURRENT_INFERENCE} inference slots are busy; retry shortly"
        )
    try:
        yield
    finally:
        _slots.release()


__all__ = [
    "INFERENCE_WAIT_SECONDS",
    "MAX_BODY_BYTES",
    "MAX_CONCURRENT_INFERENCE",
    "BodySizeLimit",
    "ServiceBusyError",
    "inference_slot",
]
