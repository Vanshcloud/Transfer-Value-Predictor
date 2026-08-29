"""The FastAPI application.

Thin on purpose. Wiring, lifespan and error handlers live here; the prediction
logic lives in :mod:`src.services.prediction`, which imports no web framework
and can be driven from a test, a batch job or a CLI without a server.

Run it with:

    uvicorn api.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import ServiceDep, build_service
from api.errors import register_error_handlers
from api.routes import router
from api.schemas import HealthResponse
from src.utils.config import load_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

VERSION = "0.1.0"

DEV_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")
"""Where the dashboard runs in development. A deployment overrides this with
CORS_ORIGINS rather than editing code."""


def settings_cors_origins() -> list[str]:
    """Allowed browser origins, from CORS_ORIGINS or the development default."""
    import os

    configured = os.environ.get("CORS_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return list(DEV_ORIGINS)


DESCRIPTION = """
Predict the market value of professional footballers, and explain every
prediction.

Two models are served, and they answer different questions:

* **performance_only** — value from on-pitch performance and biography alone.
  The scouting model: it can disagree with the market, because it has never
  been told what the market thinks.
* **with_prior_value** — forecasts how an already-known valuation will move.
  More accurate, and correspondingly less interesting.

Metrics are reported on **held-out seasons the model never saw**, which is
roughly 60% worse on EUR error than a random split would suggest and is the
number that reflects deployment.

SHAP contributions are additive in **log space**, not in euros — see
`effect_multiplier` on each contribution for the exact multiplicative reading.

The full contract is in `docs/API_CONTRACT.md`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models once, at startup.

    The older startup-event decorator is deprecated in FastAPI 0.141 and is
    not used anywhere in this service (plans/00-discovery.md section 3).
    """
    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)

    app.state.service = build_service(settings)
    logger.info("API ready; models loaded: %s", app.state.service.variants or "none")

    yield

    app.state.service = None


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so a test can construct an
    app with a substituted service without touching global state.
    """
    app = FastAPI(
        title="Transfer Value Predictor",
        description=DESCRIPTION,
        version=VERSION,
        lifespan=lifespan,
        openapi_url="/api/v1/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # The dashboard is served from a different origin in development
    # (localhost:3000 against localhost:8000), so the browser preflights every
    # request. Origins are listed rather than wildcarded: this API is
    # read-only and unauthenticated today, and "*" would be a habit that
    # becomes wrong the moment either of those changes.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(router)

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness and readiness",
        tags=["health"],
    )
    def health(service: ServiceDep) -> HealthResponse:
        """Deliberately unversioned.

        Load balancers and orchestrators should not need to know about API
        versions to decide whether a process is alive. Returns 200 even when
        degraded: the process is up, and a health check that reports down for a
        missing model would have an orchestrator restart it forever.
        """
        return HealthResponse(
            status="ok" if service.ready else "degraded",
            ready=service.ready,
            models_loaded=service.variants,
            version=VERSION,
        )

    return app


app = create_app()
