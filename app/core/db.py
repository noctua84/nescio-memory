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
    return create_engine(settings.database_url, pool_pre_ping=True)


@functools.lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    """Yields an SQLAlchemy session for database operations."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
