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
