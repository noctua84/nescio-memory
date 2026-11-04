from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from app.api.v1.router import api_router
from app.config import settings
from app.core.body_limit import BodySizeLimitMiddleware
from app.core.db import get_engine
from app.core.errors import register_exception_handlers
from app.core.log_config import configure_logging
from app.core.readiness import ReadinessReport, run_checks
from app.core.schema_checks import (
    verify_embedding_dimension,
    verify_migration_head,
    verify_pgvector_version,
)
from app.helper import get_app_version

__version__ = get_app_version()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup only. Deliberately not in create_app(): tests import this module
    # before any database exists, and a query there would fail collection.
    engine = get_engine()
    verify_pgvector_version(engine)
    verify_embedding_dimension(engine)
    verify_migration_head(engine)
    yield


def create_app() -> FastAPI:
    # First, and here rather than in the lifespan: tests never run the lifespan,
    # and anything logged while the rest of the app is wired up should already
    # use the final format. Idempotent, because tests build the app more than once.
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title=settings.app_name,
        description="NescioAI semantic memory core.",
        version=__version__,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    # Bounds /api/v1 request bytes before FastAPI buffers and parses the JSON
    # body (which otherwise happens ahead of every dependency, including
    # auth). Added as ASGI middleware, not a Depends(...), precisely so it
    # runs before dependency resolution rather than as part of it.
    app.add_middleware(BodySizeLimitMiddleware)

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health", tags=["Ops"])
    def health():
        # Liveness only — deliberately does not contact Ollama or the database,
        # so an unhealthy dependency cannot get the pod restarted.
        model = (
            settings.ollama_model
            if settings.embedding_backend == "ollama"
            else settings.local_embedding_model
        )
        return {
            "status": "ok",
            "embedding_backend": settings.embedding_backend,
            "embedding_model": model,
            # Reported because model and width together are what has to agree,
            # and #36 was a deployment where they did not. Read straight from
            # settings -- this is still the configured number, not a measured
            # one, so it stays honest about contacting nothing. The measured
            # number comes from scripts/check_embedding_dimension.py.
            "embedding_dimension": settings.embedding_dimension,
        }

    # Readiness, the counterpart to /health above. Unauthenticated on purpose
    # (an orchestrator probe carries no key) and outside the /api/v1 router.
    # Unlike /health it does contact PostgreSQL and Ollama, which is safe only
    # because run_checks bounds the wait and keeps one check per dependency in
    # flight. A sync route: the work is blocking waits on futures, which FastAPI
    # runs in its threadpool.
    @app.get(
        "/ready",
        tags=["Ops"],
        operation_id="readiness",
        summary="Readiness probe",
        description=(
            "Reports whether PostgreSQL and the embedding backend can serve, "
            "within a hard time bound. Unlike `/health` (liveness) this contacts "
            "the dependencies. 200 `ready` or 200 `degraded` (the embedding "
            "backend is down but the database works); 503 `not_ready` when the "
            "database is down. Alert when `status` is not `ready`; "
            "`checks.embedding_backend.status` names the failing dependency."
        ),
        response_model=ReadinessReport,
        responses={
            503: {
                "model": ReadinessReport,
                "description": "The database is unavailable (`not_ready`).",
            }
        },
    )
    def ready(response: Response) -> ReadinessReport:
        report = run_checks()
        if report.status == "not_ready":
            response.status_code = 503
        return report

    return app


app = create_app()
