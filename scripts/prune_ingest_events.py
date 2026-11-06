# scripts/prune_ingest_events.py
"""Delete ingest_events rows older than AUDIT_RETENTION_DAYS.

    uv run python -m scripts.prune_ingest_events

Run it on the same schedule as the database backup. Nothing else deletes these
rows, so an unscheduled table only grows. It prints how many rows went and exits
0; a database failure is an uncaught exception and a non-zero exit, so whatever
runs it sees the failure.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import timedelta

from sqlalchemy import delete, func
from sqlalchemy.orm import Session

from app.config import settings
from app.core.db import get_sessionmaker
from app.models.ingest_event import IngestEvent

EXIT_OK = 0

SessionFactory = Callable[[], AbstractContextManager[Session]]


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__.splitlines()[0])


def main(argv: Sequence[str] | None = None, session_factory: SessionFactory | None = None) -> int:
    _parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    factory = session_factory or get_sessionmaker()

    days = settings.audit_retention_days
    with factory() as db:
        result = db.execute(
            delete(IngestEvent).where(IngestEvent.at < func.now() - timedelta(days=days))
        )
        db.commit()
    print(f"pruned {result.rowcount} ingest_events rows older than {days} days")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
