"""Domain exceptions and the HTTP responses they map to.

The API layer must not know how a dependency is reached. EmbeddingBackendError is
raised by app.core.embeddings whether the backend is Ollama over HTTP or
sentence-transformers in process, so one handler covers both and no transport
detail crosses a layer boundary.

Each exception carries its own mapping as class attributes, so adding a case
means adding a class rather than extending a dispatch table.
"""
import logging

# The isinstance check below is deliberately driver-specific: SQLAlchemy
# exposes no portable "query canceled" type, so there is no
# abstraction-preserving way to detect this. If the driver ever changes
# again, this check silently stops matching, timeouts revert to the generic
# 503+Retry-After branch, and the 57014 tests below (test_error_responses.py,
# test_statement_timeout.py) start failing -- update it then.
import psycopg.errors
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DataError, OperationalError

from app.config import settings

logger = logging.getLogger(__name__)

# A dependency that is down is usually down for longer than one request, so this
# is a hint to back off rather than a promise about recovery.
RETRY_AFTER_SECONDS = 30

DATABASE_UNAVAILABLE_DETAIL = "Database unavailable"
QUERY_TIMEOUT_DETAIL = "Query exceeded time limit"
INVALID_DATA_DETAIL = "Request contains data the database cannot accept"


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
        if isinstance(exc.orig, psycopg.errors.QueryCanceled):
            logger.warning(
                "Statement canceled (SQLSTATE 57014) on %s; search "
                "statement_timeout is %d ms",
                request.url.path,
                settings.statement_timeout_ms,
            )
            return _service_unavailable(QUERY_TIMEOUT_DETAIL, retry=False)
        return _service_unavailable(DATABASE_UNAVAILABLE_DETAIL)

    @app.exception_handler(DataError)
    async def _invalid_data(request: Request, exc: DataError) -> JSONResponse:
        # DataError is SQLAlchemy's family for SQLSTATE class 22xxx (invalid text
        # representation, numeric overflow, division by zero, and similar). The
        # case that surfaced this handler is psycopg.DataError raised while
        # adapting a parameter -- e.g. a NUL byte in a text field -- which never
        # reaches Postgres at all, so this comment (deliberately) does not say
        # "Postgres rejected it". A server-side 22xxx that Postgres itself raises
        # lands here identically; both are mapped the same way below.
        #
        # 400, not 422: 422 is already owned by FastAPI/Pydantic request
        # validation, whose body is {"detail": [...]} -- a list of error objects.
        # Giving this family the same status code with a plain string detail
        # would make `detail`'s type depend on which failure produced it, which
        # is worse for a client than picking a different 4xx. 400 keeps `detail`
        # a string everywhere in this file.
        #
        # No Retry-After: unlike OperationalError's connectivity family, the
        # cause here is the submitted data itself, not a transient dependency
        # state. The same request body will fail again on every retry, so
        # advertising a retry hint would be the same lie
        # EmbeddingBackendMisconfigured avoids telling, for an unrelated reason.
        #
        # Accepted tradeoff: class 22xxx is not *exclusively* caller-caused --
        # numeric overflow or division by zero can just as easily come from this
        # application's own SQL, in which case 400 wrongly tells the caller their
        # input was bad when the bug is ours. We accept that because the values
        # that actually reach SQL in this service (repo_name, file_path, content,
        # query text, top_k) are overwhelmingly caller-supplied, and the warning
        # below is what keeps an internal-bug DataError visible to operators
        # instead of silently blamed on the client forever.
        sqlstate = getattr(exc.orig, "sqlstate", None)
        # The response body says nothing specific; the log is the only place
        # the cause survives. Without exc_info, this handler would be less
        # debuggable than the unhandled 500 it replaced, and SQLSTATE alone
        # does not identify the offending value.
        logger.warning(
            "Invalid data for the database on %s (SQLSTATE %s)",
            request.url.path,
            sqlstate or "unknown",
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": INVALID_DATA_DETAIL},
        )
