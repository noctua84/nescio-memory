"""Restore the newest backup into a scratch database, verify it, send the heartbeat.

    uv run python -m scripts.restore_check [--scheduled] [--scratch-db NAME]

Invoked with -m, not as a path: `python scripts/restore_check.py` puts scripts/
on sys.path instead of the repository root. Configuration is the environment or
.env; see the "Backup and recovery" section of the README. Needs pg_restore and
rclone on PATH, and a role on RESTORE_CHECK_DATABASE_URL that can create a
database and the vector extension. What "verified" means is in
scripts/restore_verification.py.

--scheduled is the systemd timer's mode: it refuses to run with no
BACKUP_HEARTBEAT_URL, because a dead-man's switch nobody pings is silence by
design and a backup nobody checks. BACKUP_HEARTBEAT_URL is pinged in both modes,
and only after every assertion has passed and the scratch database is gone.

Exit codes: 0 verified (and pinged, if configured), 1 failed: the backup did not
verify, a tool failed, or the ping did not land. 2 refused or invalid settings.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from scripts._backup_common import (
    DUMP_SUFFIX,
    MANIFEST_SUFFIX,
    BackupError,
    Manifest,
    RcloneRemote,
    ToolRunner,
    describe_failure,
    libpq_target,
    make_engine,
    parse_backup_name,
    run_checked,
    run_tool,
    sha256_file,
)
from scripts.backup_settings import BackupSettings, describe_settings_error
from scripts.restore_verification import VerificationError, verify_restored_database

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE_ERROR = 2

SCRATCH_PREFIX = "restore_check_"
# Fixed by default so a run killed before its cleanup is healed by the next one,
# rather than leaving a copy of everyone's notes under a name nothing returns to.
DEFAULT_SCRATCH_DB = f"{SCRATCH_PREFIX}scratch"
# Also what makes it safe to splice into DDL, which cannot take a bound name.
_SCRATCH_NAME = re.compile(rf"{SCRATCH_PREFIX}[a-z0-9_]{{1,{63 - len(SCRATCH_PREFIX)}}}")

# Fixed, under the spool, for the same reason as DEFAULT_SCRATCH_DB.
_WORK_DIR = ".restore-check"

# Silence at the switch is the alert, so a transient failure deserves a few tries
# and a dead switch must not hold up the unit. httpx applies the timeout per
# phase, so one attempt can take up to 30 s: starting none after 25 s keeps the
# total under a minute.
_HEARTBEAT_TIMEOUT_S = 10.0
_HEARTBEAT_ATTEMPTS = 3
_HEARTBEAT_BACKOFF_S = 5.0
_HEARTBEAT_LAST_START_S = 25.0


class HeartbeatError(BackupError):
    pass


class Refused(BackupError):
    """Refused before anything was touched: exit 2, not exit 1."""


def validate_scratch_db_name(name: str) -> str:
    """This script creates and DROPs the database of this name. The prefix is
    what stops a typo or a bad argument from aiming that at a real database."""
    if not _SCRATCH_NAME.fullmatch(name):
        raise Refused(
            f"refusing scratch database name {name!r}: it must start with "
            f"{SCRATCH_PREFIX!r} and contain only a-z, 0-9 and _, up to 63 characters"
        )
    return name


def create_scratch_database(admin: Engine, name: str) -> None:
    """Replace any leftover with a new, empty database. template0, so nothing an
    operator added to template1 can make the restore differ from the dump."""
    validate_scratch_db_name(name)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        connection.execute(text(f'CREATE DATABASE "{name}" TEMPLATE template0'))


def drop_scratch_database(admin: Engine, name: str) -> None:
    """Not FORCE: a session still connected means something else is using the
    database, and failing loudly beats cutting it off."""
    validate_scratch_db_name(name)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))


def fetch_newest_backup(remote: RcloneRemote, into: Path) -> tuple[Path, Manifest]:
    """Download the newest dump and its manifest, and check one against the other.

    Newest means the newest dump, even if its manifest is missing: that is a
    failed upload to report, and falling back to an older backup would keep the
    heartbeat alive while no new backup reaches the remote.
    """
    dumps = {}
    names = remote.list_names()
    for name in names:
        parsed = parse_backup_name(name)
        if parsed and parsed[1] == DUMP_SUFFIX:
            dumps[parsed[0]] = name
    if not dumps:
        raise VerificationError("no backup found on the remote")
    dump_name = dumps[max(dumps)]
    manifest_name = dump_name.removesuffix(DUMP_SUFFIX) + MANIFEST_SUFFIX
    if manifest_name not in names:
        raise VerificationError(f"{dump_name} has no manifest on the remote")

    dump_path = into / dump_name
    manifest_path = into / manifest_name
    remote.download(manifest_name, manifest_path)
    remote.download(dump_name, dump_path)

    manifest = Manifest.from_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.dump != dump_name:
        raise VerificationError(f"the manifest of {dump_name} describes another dump")
    if sha256_file(dump_path) != manifest.sha256:
        raise VerificationError(f"{dump_name} does not match its manifest's sha256")
    return dump_path, manifest


def assert_server_can_restore(admin: Engine, manifest: Manifest) -> None:
    """Say the likely reasons out loud, since pg_restore's stderr is never printed.

    The archive holds CREATE EXTENSION rather than the extension, so the server
    needs pgvector installed at the source's version or newer. pgvector is not a
    trusted extension, so the role has to be a superuser to create it (measured
    on 0.8.7: "permission denied to create extension"), unless the operator has
    marked it trusted.
    """
    with admin.connect() as connection:
        available = connection.execute(
            text("SELECT default_version FROM pg_available_extensions WHERE name = 'vector'")
        ).scalar_one_or_none()
        may_create = connection.execute(
            text(
                "SELECT rolsuper OR COALESCE((SELECT bool_or(trusted) "
                "FROM pg_available_extension_versions WHERE name = 'vector'), false) "
                "FROM pg_roles WHERE rolname = current_user"
            )
        ).scalar_one()
    if available is None:
        raise VerificationError("the restore server does not have pgvector installed")
    if _version_key(available) < _version_key(manifest.pgvector_version):
        raise VerificationError(
            f"the restore server's pgvector {available} is older than the "
            f"{manifest.pgvector_version} the dump was taken with"
        )
    if not may_create:
        raise VerificationError(
            "the RESTORE_CHECK_DATABASE_URL role cannot create the vector extension: "
            "pgvector is not a trusted extension, so the role must be a superuser"
        )


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def restore_dump(
    runner: ToolRunner, dump_path: Path | str, restore_url: str, scratch_db: str
) -> None:
    """Into a fresh empty database, so no --clean. --exit-on-error, because a
    restore that carried on past an error is not a restore to vouch for.
    --no-owner and --no-privileges, because the source's roles need not exist here.
    """
    target = libpq_target(restore_url, validate_scratch_db_name(scratch_db))
    run_checked(
        runner,
        "pg_restore",
        [
            "pg_restore",
            "--no-owner",
            "--no-privileges",
            "--exit-on-error",
            f"--dbname={target.uri}",
            str(dump_path),
        ],
        target.env,
    )


def ping_heartbeat(
    url: str,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """GET the switch, with a few bounded tries. Raises when none lands.

    The message carries no URL: the URL is the credential. Neither does the log
    of httpx, which is why main() keeps that logger above INFO.
    """
    started = clock()
    failure = ""
    with httpx.Client(transport=transport, timeout=_HEARTBEAT_TIMEOUT_S) as client:
        for attempt in range(_HEARTBEAT_ATTEMPTS):
            if attempt:
                if clock() - started + _HEARTBEAT_BACKOFF_S > _HEARTBEAT_LAST_START_S:
                    break
                sleep(_HEARTBEAT_BACKOFF_S)
            try:
                response = client.get(url)
            except httpx.HTTPError as exc:
                failure = type(exc).__name__
                continue
            if response.is_success:
                return
            failure = f"HTTP {response.status_code}"
    raise HeartbeatError(
        f"the backup verified but the heartbeat did not land ({failure}), so this "
        "run does not count as verified"
    )


@dataclass(frozen=True)
class RestoreReport:
    manifest: Manifest
    restore_seconds: float
    elapsed_seconds: float


def run_restore_check(
    settings: BackupSettings,
    *,
    scratch_db: str = DEFAULT_SCRATCH_DB,
    runner: ToolRunner = run_tool,
) -> RestoreReport:
    """Fetch, restore, verify, drop. Returns only if every step succeeded.

    The heartbeat is not here but in the caller, after this has returned, which
    is what makes "pinged only after a verified restore" structural.
    """
    validate_scratch_db_name(scratch_db)
    started = time.monotonic()
    work_dir = settings.backup_spool_dir / _WORK_DIR
    shutil.rmtree(work_dir, ignore_errors=True)
    # 0700: it is about to hold a dump of personal notes.
    work_dir.mkdir(mode=0o700)
    admin = make_engine(settings.restore_check_database_url)
    try:
        dump_path, manifest = fetch_newest_backup(
            RcloneRemote(runner, settings.backup_rclone_remote), work_dir
        )
        assert_server_can_restore(admin, manifest)
        create_scratch_database(admin, scratch_db)
        try:
            restore_started = time.monotonic()
            restore_dump(runner, dump_path, settings.restore_check_database_url, scratch_db)
            restore_seconds = time.monotonic() - restore_started

            scratch = make_engine(settings.restore_check_database_url, scratch_db)
            try:
                with Session(scratch) as db:
                    verify_restored_database(db, manifest)
            finally:
                scratch.dispose()
        finally:
            drop_scratch_database(admin, scratch_db)
    finally:
        admin.dispose()
        shutil.rmtree(work_dir, ignore_errors=True)
    return RestoreReport(manifest, restore_seconds, time.monotonic() - started)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.restore_check", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="The timer's mode: refuse to run with no BACKUP_HEARTBEAT_URL.",
    )
    parser.add_argument(
        "--scratch-db",
        default=DEFAULT_SCRATCH_DB,
        help=f"Database to create, restore into and drop. Must start with {SCRATCH_PREFIX}.",
    )
    return parser.parse_args(sys.argv[1:] if argv is None else list(argv))


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: ToolRunner = run_tool,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = _parse_args(argv)
    # httpx logs every request URL at INFO, and this URL is a credential.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        settings = BackupSettings()
    except ValidationError as exc:
        print(f"error: {describe_settings_error(exc)}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    heartbeat_url = settings.backup_heartbeat_url
    if args.scheduled and not heartbeat_url:
        print(
            "error: refusing to run --scheduled with no BACKUP_HEARTBEAT_URL: "
            "without it a failed check would alert nobody",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    try:
        report = run_restore_check(settings, scratch_db=args.scratch_db, runner=runner)
        manifest = report.manifest
        print(
            f"restore measured dump={manifest.dump} size_bytes={manifest.size_bytes} "
            f"clients={len(manifest.clients)} "
            f"notes={sum(counts.notes for counts in manifest.clients.values())} "
            f"passages={sum(counts.passages for counts in manifest.clients.values())} "
            f"pg_restore_s={report.restore_seconds:.1f} "
            f"elapsed_s={report.elapsed_seconds:.1f}",
            flush=True,
        )
        if heartbeat_url:
            ping_heartbeat(heartbeat_url, transport=transport, sleep=sleep)
    except Refused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    # Named families, each reduced by describe_failure to something safe for the
    # journal. ValidationError is the API's settings failing as the search imports
    # them; RuntimeError is the search guards' read-back. Anything else is a bug and
    # exits non-zero with its traceback, which is source lines and no values. The
    # heartbeat is unsent either way.
    except (BackupError, SQLAlchemyError, OSError, ValidationError, RuntimeError) as exc:
        print(f"error: restore check failed: {describe_failure(exc)}", file=sys.stderr)
        return EXIT_FAILED

    print(f"restore verified heartbeat={'sent' if heartbeat_url else 'not_configured'}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
