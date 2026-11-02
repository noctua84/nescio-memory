"""Make `app.*` importable without a database or a tracing backend.

Import this module BEFORE any `app.` import. The same two problems that
tests/conftest.py documents at its top apply here, for the same reasons -- but
this harness must not import from tests/, so the handling is repeated rather
than shared.

1. app/config.py declares `database_url` as a required setting and builds
   `settings` at module scope, so importing any `app.` module without
   DATABASE_URL set raises a pydantic ValidationError at import time. This
   harness never opens a connection -- it holds vectors in memory on purpose --
   but it still has to satisfy that import-time requirement.

   The placeholder points at a closed port so that if anything here ever did
   reach for a real connection, it fails immediately rather than writing to a
   developer's actual database. setdefault, not assignment: an operator who has
   exported a real DATABASE_URL keeps it, and nothing in this package will use
   it either way.

2. app/api/v1/ingest.py is wrapped in Langfuse's @observe(...), which reads its
   configuration straight from the process environment. Assigned
   unconditionally rather than via setdefault, exactly as conftest.py does and
   for the same reason: the point is to guarantee no span is exported no matter
   what the surrounding shell has set. A hand-run measurement tool has no
   business emitting traces to someone's Langfuse project.
"""
import os

APP_IMPORT_PLACEHOLDER_DSN = (
    "postgresql+psycopg://eval-placeholder:eval-placeholder@localhost:1"
    "/eval-placeholder"
)

os.environ.setdefault("DATABASE_URL", APP_IMPORT_PLACEHOLDER_DSN)
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
