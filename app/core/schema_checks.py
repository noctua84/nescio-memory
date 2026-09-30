"""Checks run once at application startup.

These exist so a configuration error fails the deploy rather than the first
request that happens to touch it.
"""
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from app.config import settings

# pgvector stores a column's dimension in pg_attribute.atttypmod.
_DIMENSION_QUERY = text(
    "SELECT atttypmod FROM pg_attribute "
    "WHERE attrelid = to_regclass('learnings') AND attname = 'embedding'"
)


def verify_embedding_dimension(engine: Engine) -> None:
    """Fail fast if EMBEDDING_DIMENSION disagrees with the learnings column.

    Without this, a mismatch is rejected by pgvector at insert time, so the
    service starts cleanly, passes its liveness probe, and then fails at the
    first ingest -- possibly long after the deploy that caused it.
    """
    try:
        with engine.connect() as connection:
            actual = connection.execute(_DIMENSION_QUERY).scalar()
    except OperationalError as exc:
        # Without this, an unreachable host surfaces as a raw SQLAlchemy
        # traceback at boot. The operator needs the database named as the
        # cause, not a stack trace through psycopg2.
        raise RuntimeError(
            "Could not connect to the database. Check DATABASE_URL and that "
            "the database is reachable."
        ) from exc

    if actual is None:
        # to_regclass returns NULL for a missing table, so the query yields no
        # row rather than raising. The remedy is migrating, not reconfiguring.
        raise RuntimeError(
            "The 'learnings' table or its 'embedding' column does not exist. "
            "Run `alembic upgrade head` before starting the application."
        )

    if actual != settings.embedding_dimension:
        raise RuntimeError(
            f"EMBEDDING_DIMENSION is {settings.embedding_dimension} but "
            f"learnings.embedding is vector({actual}). Every insert would fail. "
            "Either correct EMBEDDING_DIMENSION or migrate the column."
        )
