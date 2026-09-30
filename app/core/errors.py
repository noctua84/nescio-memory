"""Domain exceptions and the HTTP responses they map to.

The API layer must not know how a dependency is reached. EmbeddingBackendError is
raised by app.core.embeddings whether the backend is Ollama over HTTP or
sentence-transformers in process, so one handler covers both and no transport
detail crosses a layer boundary.

Each exception carries its own mapping as class attributes, so adding a case
means adding a class rather than extending a dispatch table.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

# A dependency that is down is usually down for longer than one request, so this
# is a hint to back off rather than a promise about recovery.
RETRY_AFTER_SECONDS = 30

DATABASE_UNAVAILABLE_DETAIL = "Database unavailable"


class EmbeddingBackendError(RuntimeError):
    """The configured embedding backend could not produce a vector.

    Transient by assumption: the backend was reachable in principle and may work
    on a later attempt.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Embedding backend unavailable"
    retry_after = True


class EmbeddingBackendBadResponse(EmbeddingBackendError):
    """The backend answered, but not with an embedding we could read."""

    detail = "Embedding backend returned an unexpected response"


class EmbeddingBackendMisconfigured(EmbeddingBackendError):
    """The backend cannot work as deployed.

    Distinct from its parent because retrying will never help: the remedy is a
    deployment change, not patience. It therefore maps to 500 and sends no
    Retry-After, since advertising one would tell the client something untrue.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Embedding backend is misconfigured"
    retry_after = False


def _service_unavailable(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": detail},
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Map dependency failures onto HTTP responses. Called from create_app()."""

    @app.exception_handler(EmbeddingBackendError)
    async def _embedding_backend_failed(
        request: Request, exc: EmbeddingBackendError
    ) -> JSONResponse:
        # detail comes from the class, never from str(exc): exception messages
        # carry internal detail for logs and must not reach the client.
        headers = (
            {"Retry-After": str(RETRY_AFTER_SECONDS)} if exc.retry_after else None
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    @app.exception_handler(OperationalError)
    async def _database_unavailable(
        request: Request, exc: OperationalError
    ) -> JSONResponse:
        # SQLAlchemy is already the abstraction over the database, so handling its
        # exception directly is correct rather than leaky. OperationalError is the
        # connectivity family; programming errors are bugs and stay 500s.
        return _service_unavailable(DATABASE_UNAVAILABLE_DETAIL)
