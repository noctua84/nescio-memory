import functools
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings


# Lazy on purpose (issue #16): create_engine() imports the DBAPI driver module,
# so building the engine at import time means a bad DATABASE_URL/driver crashes
# every entry point at import, including export_openapi.py, which never
# touches the database. Deferring to first use keeps import side-effect-free.
@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        # Issue #30: configurable connect timeout prevents hanging on unreachable
        # Postgres instances. With pool_pre_ping, a dead database adds this
        # timeout inside request processing.
        connect_args={"connect_timeout": settings.db_connect_timeout_s},
        # Hide bound SQL parameters: exception strings reach Langfuse span
        # status_message and logs, and bound parameters include note content.
        hide_parameters=True,
    )


@functools.lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    """Yields an SQLAlchemy session for database operations.

    THE INVARIANT: a request's session holds a connection only while a thread is
    running on it, never across a thread hop. Whatever a dependency or a route
    reads or writes, it ends its transaction (commit) before it returns, and the
    next one begins its own. The other files point here rather than restate it.

    Why. FastAPI runs every sync dependency, the sync route and the response-model
    validation as separate tasks on anyio's one process-wide pool of 40 threads,
    each taking a token from a FIFO limiter. A session keeps its connection
    checked out from its first statement until commit/close, and a request between
    two such tasks holds that connection while holding no thread at all. So the
    connection pool does not bound the holders: any number of requests can sit
    between hops with a connection each, queued for a thread, while the requests
    that did get threads block in pool checkout waiting for one of those
    connections. Nothing in that cycle can move, and only the pool's 30 s
    pool_timeout breaks it, as sqlalchemy.exc.TimeoutError (a 503, see
    app/core/errors.py). Measured on 200 concurrent searches from one client:
    unchanged, it stalled for 120 s and more in every run; with a pool of 40 it
    still stalled 91 s; releasing in auth alone, 61 s; releasing in auth and at the
    end of the route, every run finished in 2-3 s with no error.

    There are two places a request crosses a hop with its session in use, and
    they are the only two that matter:

    1. From get_current_client to the next dependency or the route
       (app/core/security.py).
    2. From the route's return to response validation. A route that has only
       read must commit before returning (app/api/v1/search.py,
       app/api/v1/notes.py get_note); one that wrote already commits as its
       single transaction boundary.

    Not a hop: the exit of this dependency. FastAPI runs a yield dependency's
    cleanup outside the limiter, so the close below needs no thread token, and an
    error raised anywhere in the request reaches it the same way.

    Within one task the invariant allows a transaction as long as the task needs
    it. Ingest keeps one open across embedding on purpose (see ingest.py), because
    its route is a single task and so never crosses a hop mid-transaction.
    """
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
