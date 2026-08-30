"""One error envelope, so a client writes one error path.

FastAPI's defaults return three different shapes — ``{"detail": "..."}`` for an
HTTPException, a list for a validation error, and an HTML page for an unhandled
exception. A consumer then needs three parsers. These handlers collapse all of
them into the shape documented in docs/API_CONTRACT.md section 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.services.prediction import (
    InvalidFeaturesError,
    ModelNotFoundError,
    ModelUnavailableError,
    PlayerNotFoundError,
    SeasonNotFoundError,
    ServiceError,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Which service failure becomes which status code. Kept as data so the mapping
# can be read in one glance and matches the contract table line for line.
STATUS_FOR: dict[type[ServiceError], int] = {
    PlayerNotFoundError: status.HTTP_404_NOT_FOUND,
    SeasonNotFoundError: status.HTTP_404_NOT_FOUND,
    ModelNotFoundError: status.HTTP_404_NOT_FOUND,
    InvalidFeaturesError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ModelUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


CODE_FOR: dict[int, str] = {
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_error",
}
"""Codes for the failures Starlette raises before this application sees them."""


def error_response(
    code: str, message: str, status_code: int, detail: object = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "detail": detail}},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach the handlers that produce the documented envelope."""

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        status_code = STATUS_FOR.get(type(exc), status.HTTP_400_BAD_REQUEST)
        return error_response(exc.code, exc.message, status_code, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # The field-level errors are the entire value of a 422. A bare
        # "Unprocessable Entity" forces the client to guess what was wrong.
        return error_response(
            "validation_error",
            "the request body failed validation",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_serialisable(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # 404s for unrouted paths and 405s for the wrong method are raised by
        # Starlette itself, never by this code, so without this they come back
        # as {"detail": "Not Found"} — a second shape for the same client to
        # parse. CODE_FOR names the common ones; anything else gets a code
        # derived from the status rather than a fabricated one.
        code = CODE_FOR.get(exc.status_code, f"http_{exc.status_code}")
        return error_response(code, str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        """The envelope of last resort.

        This module's whole claim is that *every* non-2xx response has one
        shape. Without this handler that claim is false for exactly the case
        where a client can least afford to guess: FastAPI's default 500 is
        ``text/plain`` "Internal Server Error", so a consumer parsing JSON gets
        an exception while handling an exception.

        The message is deliberately fixed. An unhandled exception's text can
        carry a file path, a query or a fragment of data, and this endpoint is
        unauthenticated; the detail goes to the log, where it belongs, and the
        client gets a code it can branch on.
        """
        logger.exception("unhandled error serving a request: %s", exc)
        return error_response(
            "internal_error",
            "the server failed to handle this request",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _serialisable(errors: Sequence[Any]) -> list[dict[str, object]]:
    """Drop what will not serialise.

    Pydantic puts the offending input, and sometimes the exception object
    itself, into `ctx`. Neither is reliably JSON-able, and a 500 raised while
    rendering a 422 is a genuinely confusing failure to debug.
    """
    return [
        {key: value for key, value in error.items() if key in {"type", "loc", "msg"}}
        for error in errors
    ]
