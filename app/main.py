from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.config import settings
from app.core.db import get_engine
from app.core.errors import register_exception_handlers
from app.core.schema_checks import verify_embedding_dimension, verify_pgvector_version
from app.helper import get_app_version

__version__ = get_app_version()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup only. Deliberately not in create_app(): tests import this module
    # before any database exists, and a query there would fail collection.
    engine = get_engine()
    verify_pgvector_version(engine)
    verify_embedding_dimension(engine)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="NescioAI semantic memory core.",
        version=__version__,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

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

    return app


app = create_app()
