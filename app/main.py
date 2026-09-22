from fastapi import FastAPI

from app.api.v1.router import api_router
from app.config import settings
from app.helper import get_app_version

__version__ = get_app_version()

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="NescioAI semantic memory core.",
        version=__version__,
    )

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
        }

    return app


app = create_app()