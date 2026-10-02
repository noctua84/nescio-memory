"""Domain exceptions and the HTTP responses they map to.

The API layer must not know how a dependency is reached. EmbeddingBackendError is
raised by app.core.embeddings whether the backend is Ollama over HTTP or
sentence-transformers in process, so one handler covers both and no transport
detail crosses a layer boundary.

Each exception carries its own mapping as class attributes, so adding a case
means adding a class rather than extending a dispatch table.
"""
import logging

# The isinstance check below is deliberately driver-specific: a hand-built
# QueryCanceled has no pgcode, and SQLAlchemy exposes no portable "query
# canceled" type, so there is no abstraction-preserving way to detect this.
# If the driver ever changes to psycopg3, this check silently stops matching
# and timeouts revert to the generic 503+Retry-After branch -- update it then.
import psycopg2.errors
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from app.config import settings

logger = logging.getLogger(__name__)

# A dependency that is down is usually down for longer than one request, so this
# is a hint to back off rather than a promise about recovery.
RETRY_AFTER_SECONDS = 30

DATABASE_UNAVAILABLE_DETAIL = "Database unavailable"
QUERY_TIMEOUT_DETAIL = "Query exceeded time limit"


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


def _service_unavailable(detail: str, retry: bool = True) -> JSONResponse:
    headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if retry else None
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": detail},
        headers=headers,
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
        #
        # QueryCanceled (SQLSTATE 57014) is a special case within that family.
        # Only the search transaction ever sets a statement_timeout (see
        # LearningRepository.search), so in practice 57014 here is almost
        # always that timeout firing -- a deterministic outcome for the same
        # input, not a transient outage. But 57014 is not exclusively a
        # statement_timeout: an admin's pg_cancel_backend() or a role-level
        # statement_timeout would raise the identical SQLSTATE, and the
        # response mapping deliberately treats all of these alike (accepted
        # decision), so the log below must not claim a cause it cannot know
        # and states only what is actually known: where and what the
        # configured limit is.
        if isinstance(exc.orig, psycopg2.errors.QueryCanceled):
            logger.warning(
                "Statement canceled (SQLSTATE 57014) on %s; search "
                "statement_timeout is %d ms",
                request.url.path,
                settings.statement_timeout_ms,
            )
            return _service_unavailable(QUERY_TIMEOUT_DETAIL, retry=False)
        return _service_unavailable(DATABASE_UNAVAILABLE_DETAIL)
