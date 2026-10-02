"""Importing app.main must not build a SQLAlchemy engine (issue #16).

create_engine() imports the DBAPI driver module, so an eager module-level
engine crashes every entry point at import if the configured driver is not
installed -- including export_openapi.py, which never touches the database.
These tests prove import (and OpenAPI generation, which only walks routes and
Pydantic models) stays driver-free, and that the lazy engine is still cached
once something does ask for it.

The import check runs in a subprocess: tests/conftest.py already imports
app.main in-process (to build its TestClient fixtures), so by the time this
test module runs, app.core.db.get_engine has already had the chance to be
called in-process by other tests/fixtures. A fresh interpreter is the only
way to observe "nothing touched create_engine during import".
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# pg8000 is a real SQLAlchemy dialect that is deliberately not installed on
# this branch (see uv.lock) -- if create_engine() were ever called eagerly,
# constructing the dialect would raise ModuleNotFoundError at import time.
UNINSTALLED_DRIVER_URL = "postgresql+pg8000://x:x@localhost:1/x"

SUBPROCESS_CODE = """
import sqlalchemy

calls = []
_real_create_engine = sqlalchemy.create_engine

def _spy(*args, **kwargs):
    calls.append((args, kwargs))
    return _real_create_engine(*args, **kwargs)

sqlalchemy.create_engine = _spy
sqlalchemy.engine.create_engine = _spy

import app.main

app.main.app.openapi()

assert not calls, f"create_engine was called during import/openapi(): {calls}"

import app.core.db as db_module

assert db_module.get_engine.cache_info().currsize == 0, (
    "get_engine's lru_cache should still be empty -- nothing in import or "
    "openapi() generation should have called it"
)

print("OK")
"""


def test_importing_app_main_does_not_create_an_engine():
    """Import + OpenAPI generation must not need the configured DBAPI driver."""
    env = os.environ.copy()
    env["DATABASE_URL"] = UNINSTALLED_DRIVER_URL
    env["LANGFUSE_TRACING_ENABLED"] = "false"

    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_CODE],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"subprocess failed (returncode={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_get_engine_is_cached():
    """get_engine() must return the same Engine object on repeated calls.

    create_engine() does not connect, and the placeholder URL set in
    conftest.py uses psycopg (v3), which IS installed, so calling get_engine()
    in-process here is safe and does not touch a real database.
    """
    from app.core.db import get_engine

    get_engine.cache_clear()
    try:
        first = get_engine()
        second = get_engine()
        assert first is second
    finally:
        get_engine.cache_clear()
