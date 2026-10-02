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
        # cause, not a stack trace through psycopg.
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


# pgvector reports its own version through pg_extension. hnsw.iterative_scan,
# which LearningRepository.search sets on every query, arrived in 0.8.0.
_PGVECTOR_VERSION_QUERY = text(
    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
)

_MINIMUM_PGVECTOR = (0, 8)


def verify_pgvector_version(engine: Engine) -> None:
    """Fail fast if pgvector predates the settings that search depends on.

    The `vector` extension reserves the `hnsw` GUC prefix, and PostgreSQL
    rejects an unknown parameter under a reserved prefix once the library is
    loaded in that backend. On pgvector 0.7.x the set_config call in
    LearningRepository.search therefore raises and every search returns 500.
    Checking at boot turns that into a failed deploy instead.
    """
    try:
        with engine.connect() as connection:
            version = connection.execute(_PGVECTOR_VERSION_QUERY).scalar()
    except OperationalError as exc:
        raise RuntimeError(
            "Could not connect to the database. Check DATABASE_URL and that "
            "the database is reachable."
        ) from exc

    if version is None:
        raise RuntimeError(
            "The 'vector' extension is not installed. "
            "Run `alembic upgrade head` before starting the application."
        )

    # extversion looks like "0.8.6". Only major.minor matters for a floor check,
    # and a non-numeric suffix on the patch component is not worth parsing.
    try:
        parsed = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        raise RuntimeError(
            f"Could not parse the installed pgvector version {version!r}. "
            "This application requires pgvector >= 0.8.0."
        ) from None

    if parsed < _MINIMUM_PGVECTOR:
        raise RuntimeError(
            f"pgvector {version} is installed but this application requires "
            ">= 0.8.0. Search sets hnsw.iterative_scan, which does not exist "
            "before 0.8.0, and PostgreSQL rejects unknown parameters under the "
            "reserved 'hnsw' prefix, so every search would fail. Upgrade the "
            "extension."
        )
