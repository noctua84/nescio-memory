"""Checks run once at application startup.

These exist so a configuration error fails the deploy rather than the first
request that happens to touch it.
"""
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from app.config import settings


def verify_embedding_dimension(engine: Engine) -> None:
    """Fail fast if EMBEDDING_DIMENSION disagrees with the passages.embedding column.

    Without this, a mismatch is rejected by pgvector at insert time, so the
    service starts cleanly, passes its liveness probe, and then fails at the
    first ingest -- possibly long after the deploy that caused it.
    """
    tables = ("passages",)

    try:
        with engine.connect() as connection:
            for table in tables:
                query = text(
                    f"SELECT atttypmod FROM pg_attribute "
                    f"WHERE attrelid = to_regclass('{table}') AND attname = 'embedding'"
                )
                actual = connection.execute(query).scalar()

                if actual is None:
                    # to_regclass returns NULL for a missing table, so the query yields no
                    # row rather than raising. The remedy is migrating, not reconfiguring.
                    raise RuntimeError(
                        f"The '{table}' table or its 'embedding' column does not exist. "
                        "Run `alembic upgrade head` before starting the application."
                    )

                if actual != settings.embedding_dimension:
                    raise RuntimeError(
                        f"EMBEDDING_DIMENSION is {settings.embedding_dimension} but "
                        f"{table}.embedding is vector({actual}). Every insert would fail. "
                        "Either correct EMBEDDING_DIMENSION or migrate the column."
                    )
    except OperationalError as exc:
        # Without this, an unreachable host surfaces as a raw SQLAlchemy
        # traceback at boot. The operator needs the database named as the
        # cause, not a stack trace through psycopg.
        raise RuntimeError(
            "Could not connect to the database. Check DATABASE_URL and that "
            "the database is reachable."
        ) from exc


# pgvector reports its own version through pg_extension. hnsw.iterative_scan,
# which PassageSearchRepository.search sets on every query, arrived in 0.8.0.
_PGVECTOR_VERSION_QUERY = text(
    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
)

_MINIMUM_PGVECTOR = (0, 8)


def verify_pgvector_version(engine: Engine) -> None:
    """Fail fast if pgvector predates the settings that search depends on.

    The `vector` extension reserves the `hnsw` GUC prefix, and PostgreSQL
    rejects an unknown parameter under a reserved prefix once the library is
    loaded in that backend. On pgvector 0.7.x the set_config call in
    PassageSearchRepository.search therefore raises and every search returns 500.
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


# Resolved from this file, not the working directory: the service may be started
# from anywhere, and alembic.ini's script_location is itself relative to the ini
# (%(here)s/alembic), so locating the ini is enough to locate the migrations.
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"

_ALEMBIC_VERSION_TABLE_QUERY = text("SELECT to_regclass('alembic_version') IS NOT NULL")
_ALEMBIC_VERSION_QUERY = text("SELECT version_num FROM alembic_version")


def _script_heads() -> set[str]:
    if not _ALEMBIC_INI.is_file():
        raise RuntimeError(
            f"Could not find {_ALEMBIC_INI} to read the expected migration "
            "revision. The application must run from a checkout that includes "
            "alembic.ini and the alembic/ directory."
        )
    return set(ScriptDirectory.from_config(Config(str(_ALEMBIC_INI))).get_heads())


def verify_migration_head(engine: Engine) -> None:
    """Fail fast if the database is not at the code's Alembic head.

    The other startup checks only probe individual objects (the embedding
    column, the pgvector version), so a database that is merely behind on
    migrations -- a new column or index not yet applied -- would start cleanly
    and fail on the first request that touches it. Comparing revisions turns
    that into a failed deploy that names the remedy.
    """
    expected = _script_heads()

    try:
        with engine.connect() as connection:
            # Probed first because selecting from a missing alembic_version
            # would raise; an unmigrated database should report "no revision".
            has_table = connection.execute(_ALEMBIC_VERSION_TABLE_QUERY).scalar()
            current = (
                set(connection.execute(_ALEMBIC_VERSION_QUERY).scalars().all())
                if has_table
                else set()
            )
    except OperationalError as exc:
        raise RuntimeError(
            "Could not connect to the database. Check DATABASE_URL and that "
            "the database is reachable."
        ) from exc

    if current == expected:
        return

    current_text = ", ".join(sorted(current)) or "none (no alembic_version row)"
    expected_text = ", ".join(sorted(expected))
    raise RuntimeError(
        f"The database is at migration revision {current_text} but this "
        f"application expects {expected_text}. "
        "Run `alembic upgrade head` before starting the application. If the "
        "database revision is newer than the code, deploy the matching code "
        "instead."
    )
