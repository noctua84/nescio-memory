"""Regression test for database URL driver resolution (issue #16).

SQLAlchemy 2.1 resolves a bare `postgresql://` URL to the psycopg (v3) dialect.
The project previously depended only on psycopg2, causing a ModuleNotFoundError
when create_engine() imported the missing driver. This test ensures the fix
(switching to psycopg 3 and using explicit `postgresql+psycopg://` URLs) is not
regressed: it verifies create_engine can resolve driver URLs and produces the
correct driver name.

The test does not connect: create_engine() imports the DBAPI module, which fails
if the driver is not installed. This is the same failure surface as issue #16.
"""
from pathlib import Path

import pytest
import sqlalchemy


def _read_env_example_database_url():
    """Extract DATABASE_URL from .env.example in the project root."""
    env_example = Path(__file__).resolve().parent.parent / ".env.example"
    with open(env_example) as f:
        for line in f:
            if line.startswith("DATABASE_URL="):
                # Strip the key, line ending, and any trailing whitespace
                return line[len("DATABASE_URL="):].rstrip()
    raise ValueError(f"DATABASE_URL not found in {env_example}")


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://user:password@localhost:5432/nescio_memory",  # Bare URL: the exact #16 case
        _read_env_example_database_url(),  # From .env.example
    ],
    ids=["bare_postgresql_url", "env_example_url"],
)
def test_database_url_resolves_to_psycopg_driver(database_url):
    """Verify create_engine can resolve a database URL to the psycopg driver.

    Issue #16: SQLAlchemy 2.1 resolves bare `postgresql://` to psycopg (v3),
    but the project then had only psycopg2 installed, causing ModuleNotFoundError.
    This test ensures the installed driver matches the resolved dialect.
    """
    engine = sqlalchemy.create_engine(database_url)
    try:
        assert engine.dialect.driver == "psycopg", (
            f"Expected psycopg driver for {database_url!r}, "
            f"but got {engine.dialect.driver!r}"
        )
    finally:
        engine.dispose()
