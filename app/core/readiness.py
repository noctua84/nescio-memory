"""Readiness checks behind GET /ready.

/health is liveness and contacts nothing. This module is the other half: it
asks PostgreSQL and the embedding backend whether they can serve, under two
guarantees the endpoint needs because it is unauthenticated and polled:

1. A hard time bound. The response comes back within the configured per-
   dependency timeout (plus a little overhead) even if a dependency hangs.
2. At most one check in flight per dependency. A hung dependency therefore
   holds exactly one worker thread, no matter how many probes arrive; probes
   never queue up behind it and never open one connection each.

Nothing the dependency says reaches the response body -- no exception text, no
host names. The detail goes to the log at WARNING instead.
"""
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Literal

import httpx
import psycopg.errors
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.core.db import get_engine

logger = logging.getLogger(__name__)

# Suffix that app.config guarantees OLLAMA_URL ends with when the backend is
# "ollama"; stripping it yields the server base that /api/show lives under.
_OLLAMA_EMBED_SUFFIX = "/api/embed"


class DatabaseCheck(BaseModel):
    status: Literal["ok", "fail", "timeout"] = Field(
        description="`ok`: a trivial query succeeded. `fail`: the database could "
        "not be reached or the query errored. `timeout`: no answer within "
        "READY_DB_TIMEOUT_S."
    )


class EmbeddingBackendCheck(BaseModel):
    status: Literal["ok", "fail", "timeout", "not_checked"] = Field(
        description="`ok`: Ollama answered and the configured model is pulled. "
        "`fail`: unreachable, or the model is missing. `timeout`: no answer "
        "within READY_EMBEDDING_TIMEOUT_S. `not_checked`: the `local` backend "
        "runs in-process, so there is nothing remote to probe."
    )


class ReadinessChecks(BaseModel):
    database: DatabaseCheck
    embedding_backend: EmbeddingBackendCheck


class ReadinessReport(BaseModel):
    status: Literal["ready", "degraded", "not_ready"] = Field(
        description="`ready`: everything is fine. `degraded`: the database works "
        "but the embedding backend does not (HTTP 200, so replicas stay in "
        "rotation; alert on this field). `not_ready`: the database is down "
        "(HTTP 503)."
    )
    checks: ReadinessChecks


def _failure(dependency: str, exc: BaseException) -> None:
    # Class name and message only, no traceback: a probe repeats every few
    # seconds and a dead dependency would otherwise print a stack each time.
    # Hide-parameters on the engine keeps bound values out of SQLAlchemy's text.
    logger.warning(
        "Readiness check failed: dependency=%s error=%s: %s",
        dependency,
        type(exc).__name__,
        exc,
    )


def check_database(timeout_s: float) -> str:
    """``SELECT 1`` under a statement timeout; returns ok | fail | timeout."""
    # set_config(..., true) is SET LOCAL as a bindable function call: it lasts
    # for this transaction only, so the pooled connection goes back unchanged.
    # At least 1 ms: 0 would disable the timeout in Postgres.
    timeout_ms = max(1, int(timeout_s * 1000))
    try:
        with get_engine().connect() as connection:
            connection.execute(
                text("SELECT set_config('statement_timeout', :ms, true)"),
                {"ms": str(timeout_ms)},
            )
            connection.execute(text("SELECT 1"))
    except OperationalError as exc:
        # SQLAlchemy wraps the driver error; .orig is the psycopg exception.
        _failure("database", exc)
        if isinstance(exc.orig, psycopg.errors.QueryCanceled):
            return "timeout"
        return "fail"
    except Exception as exc:
        _failure("database", exc)
        return "fail"
    return "ok"


def check_embedding_backend(timeout_s: float) -> str:
    """Ask Ollama whether the model is pulled; returns ok | fail | timeout |
    not_checked."""
    if settings.embedding_backend != "ollama":
        return "not_checked"
    # /api/show rather than /api/embed: it proves the server is up and the model
    # is pulled without loading the model into memory or running an embedding.
    base = settings.ollama_url[: -len(_OLLAMA_EMBED_SUFFIX)]
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                f"{base}/api/show", json={"model": settings.ollama_model}
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        _failure("embedding_backend", exc)
        return "timeout"
    except Exception as exc:
        _failure("embedding_backend", exc)
        return "fail"
    return "ok"


class _SingleFlightProbe:
    """Runs one dependency check at a time and shares its result.

    Per request (``begin``):

    * a result younger than the TTL is reused;
    * otherwise, if a check is in flight, the caller waits on that same future
      and nothing new is submitted;
    * otherwise exactly one check is submitted.

    The check runs on this probe's own single-thread executor and the worker
    records the result and clears the in-flight marker itself, under the lock.
    A check is only submitted when none is in flight, so the executor's queue
    can never hold more than the one running item: a hung dependency costs one
    thread, and ``begin`` is never blocked by it.
    """

    def __init__(
        self,
        dependency: str,
        check: Callable[[float], str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dependency = dependency
        self._check = check
        self._clock = clock
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"ready-{dependency}"
        )
        self._inflight: Future[str] | None = None
        self._last: str | None = None
        self._last_at = 0.0

    def begin(self, check_timeout_s: float, ttl_s: float) -> "str | Future[str]":
        """A finished result to use now, or the future that will produce it."""
        with self._lock:
            if (
                ttl_s > 0
                and self._last is not None
                and self._clock() - self._last_at < ttl_s
            ):
                return self._last
            if self._inflight is not None:
                return self._inflight
            # Submitted and published under the lock: the worker needs the same
            # lock to clear it, so it cannot finish "before" it was recorded.
            self._inflight = self._executor.submit(self._run, check_timeout_s)
            return self._inflight

    def _run(self, check_timeout_s: float) -> str:
        result = "fail"
        try:
            result = self._check(check_timeout_s)
        except Exception as exc:  # a check should not raise, but never wedge
            _failure(self.dependency, exc)
        finally:
            # finally, so even a BaseException cannot leave the marker set and
            # turn every later request into a permanent "timeout".
            with self._lock:
                self._last = result
                self._last_at = self._clock()
                self._inflight = None
        return result

    def shutdown(self) -> None:
        """Wait for the worker to finish. Used by tests, never by the app."""
        self._executor.shutdown(wait=True)


# Late-bound lambdas: the probe holds the lambda, the lambda looks the check up
# in this module at call time, so a test that replaces `check_database` (or
# swaps the probe) is seen by the endpoint.
DATABASE_PROBE = _SingleFlightProbe("database", lambda t: check_database(t))
EMBEDDING_PROBE = _SingleFlightProbe(
    "embedding_backend", lambda t: check_embedding_backend(t)
)


def _resolve(handle: "str | Future[str]", deadline: float, dependency: str) -> str:
    if isinstance(handle, str):
        return handle
    try:
        return handle.result(timeout=max(0.0, deadline - time.monotonic()))
    except FutureTimeout:
        # Not recorded as a result and nothing new is submitted: the check keeps
        # running in the background, bounded by the connect / httpx timeout.
        logger.warning(
            "Readiness check still running at its deadline: dependency=%s",
            dependency,
        )
        return "timeout"


def run_checks() -> ReadinessReport:
    started = time.monotonic()
    # Both probes are started before either is waited on, and each deadline is
    # measured from the request start, so the total is bounded by the larger
    # timeout rather than their sum.
    ttl = settings.ready_cache_ttl_s
    db_handle = DATABASE_PROBE.begin(settings.ready_db_timeout_s, ttl)
    embedding_handle = EMBEDDING_PROBE.begin(settings.ready_embedding_timeout_s, ttl)

    database = _resolve(
        db_handle, started + settings.ready_db_timeout_s, DATABASE_PROBE.dependency
    )
    embedding = _resolve(
        embedding_handle,
        started + settings.ready_embedding_timeout_s,
        EMBEDDING_PROBE.dependency,
    )

    if database != "ok":
        overall = "not_ready"
    elif embedding in ("fail", "timeout"):
        # Ollama is shared by every replica: failing readiness on it would eject
        # all of them at once and turn a degraded search into a total outage.
        overall = "degraded"
    else:
        overall = "ready"

    return ReadinessReport.model_validate(
        {
            "status": overall,
            "checks": {
                "database": {"status": database},
                "embedding_backend": {"status": embedding},
            },
        }
    )
