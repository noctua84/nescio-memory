"""What scripts/backup.py and scripts/restore_check.py share: backup names, the
manifest, the facts read from a database, and the external tools.

Neither script imports app.config for any of this, so the manifest is produced
and checked by the same SQL without the API's settings.

The external tools run through an injectable ToolRunner. A tool's stderr is
captured and never printed, because it is not safe to: pg_restore quotes the
failing COPY line, which is note content, and rclone quotes paths. A failure is
reported as the step and its exit code; the README runbook gives the command to
repeat by hand when the cause is needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

DUMP_SUFFIX = ".dump"
MANIFEST_SUFFIX = ".manifest.json"

_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
# Flat and short: rclone crypt encrypts each path segment and most stores cap an
# encrypted segment, which leaves a plaintext name about 143 characters.
_BACKUP_NAME = re.compile(r"nescio-(\d{8}T\d{6}Z)(\.dump|\.manifest\.json)")

_CONNECT_TIMEOUT_S = 10
_READ_CHUNK_BYTES = 1024 * 1024


class BackupError(Exception):
    """A backup or restore check failed. The message is safe to print."""


class ToolError(BackupError):
    def __init__(self, step: str, returncode: int) -> None:
        super().__init__(f"{step} failed with exit code {returncode}")
        self.step = step
        self.returncode = returncode


def backup_stem(moment: datetime) -> str:
    """The name both files of one backup start with, stamped in UTC."""
    return f"nescio-{moment.astimezone(UTC).strftime(_STAMP_FORMAT)}"


def parse_backup_name(name: str) -> tuple[datetime, str] | None:
    """(moment, suffix) for a file this tool wrote, None for anything else.

    Anything else is left alone by pruning: only a name this tool could have
    written may ever be deleted.
    """
    match = _BACKUP_NAME.fullmatch(name)
    if match is None:
        return None
    try:
        moment = datetime.strptime(match.group(1), _STAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None
    return moment, match.group(2)


@dataclass(frozen=True)
class ToolResult:
    returncode: int
    stdout: str


class ToolRunner(Protocol):
    def __call__(
        self, args: Sequence[str], env: Mapping[str, str] | None = None
    ) -> ToolResult: ...


def run_tool(args: Sequence[str], env: Mapping[str, str] | None = None) -> ToolResult:
    """Run a tool to completion, keeping stderr out of every output stream.

    No timeout here: a dump takes as long as the data does, so the bound is the
    CronJob's activeDeadlineSeconds, which kills the whole pod.
    """
    try:
        completed = subprocess.run(
            list(args),
            env={**os.environ, **(env or {})},
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        # The shell's "command not found", so the journal shows what is missing.
        return ToolResult(127, "")
    return ToolResult(completed.returncode, completed.stdout)


def run_checked(
    runner: ToolRunner,
    step: str,
    args: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> str:
    """Any non-zero exit is a failure, rclone's "partial" codes included."""
    result = runner(args, env)
    if result.returncode != 0:
        raise ToolError(step, result.returncode)
    return result.stdout


@dataclass(frozen=True)
class LibpqTarget:
    """A database for pg_dump and pg_restore.

    The password travels in the environment: argv is world-readable in /proc and
    `ps`, the environment of another user's process is not.
    """

    uri: str
    env: Mapping[str, str]


def libpq_target(url: str, database: str | None = None) -> LibpqTarget:
    parsed = make_url(url)
    if database is not None:
        parsed = parsed.set(database=database)
    # URL.create, not parsed.set(password=None): set() treats None as "leave as
    # is" and the password would land in argv. drivername: libpq does not know
    # SQLAlchemy's +psycopg suffix.
    bare = URL.create(
        "postgresql",
        username=parsed.username,
        host=parsed.host,
        port=parsed.port,
        database=parsed.database,
        query=parsed.query,
    )
    env = {"PGCONNECT_TIMEOUT": str(_CONNECT_TIMEOUT_S)}
    if parsed.password is not None:
        env["PGPASSWORD"] = str(parsed.password)
    return LibpqTarget(uri=bare.render_as_string(hide_password=False), env=env)


def make_engine(url: str, database: str | None = None) -> Engine:
    """An engine for one short job: no pool, so nothing outlives dispose() and
    keeps a DROP DATABASE from succeeding. hide_parameters for the reason
    app/core/db.py gives: bound parameters include note content and vectors.
    """
    parsed = make_url(url)
    if database is not None:
        parsed = parsed.set(database=database)
    return create_engine(
        parsed,
        poolclass=NullPool,
        hide_parameters=True,
        connect_args={"connect_timeout": _CONNECT_TIMEOUT_S},
    )


class RcloneRemote:
    """The operations this tool needs on one remote, and no others.

    There is deliberately no `sync`: a sync from a spool that went missing would
    delete the off-host copies. Each upload names one file and each delete names
    one file.
    """

    def __init__(self, runner: ToolRunner, remote: str) -> None:
        self._runner = runner
        self._remote = remote.rstrip("/")
        self._name = remote.split(":", 1)[0]

    def assert_crypt(self) -> None:
        """Refuse any remote that is not rclone crypt: the dumps hold personal
        notes, and a plain S3 remote would accept them just as happily.

        `listremotes --long` prints name and type only, never a credential.
        """
        listing = run_checked(
            self._runner, "rclone listremotes", ["rclone", "listremotes", "--long"]
        )
        for line in listing.splitlines():
            name, separator, rest = line.partition(":")
            if separator and name == self._name:
                if rest.split()[:1] == ["crypt"]:
                    return
                raise BackupError(
                    "BACKUP_RCLONE_REMOTE must be an rclone crypt remote; "
                    "refusing to store personal notes on any other type"
                )
        raise BackupError("BACKUP_RCLONE_REMOTE names a remote rclone does not know")

    def _path(self, name: str) -> str:
        return f"{self._remote}/{name}"

    def list_names(self) -> list[str]:
        """Decrypted file names: listing through the crypt remote decrypts them."""
        listing = run_checked(
            self._runner,
            "rclone lsjson",
            ["rclone", "lsjson", "--files-only", self._remote],
        )
        try:
            return [entry["Name"] for entry in json.loads(listing)]
        except (ValueError, TypeError, KeyError):
            raise BackupError("rclone lsjson printed something unreadable") from None

    def upload(self, local: Path, name: str) -> None:
        run_checked(
            self._runner, "rclone copyto (upload)",
            ["rclone", "copyto", str(local), self._path(name)],
        )

    def download(self, name: str, local: Path) -> None:
        run_checked(
            self._runner, "rclone copyto (download)",
            ["rclone", "copyto", self._path(name), str(local)],
        )

    def delete(self, name: str) -> None:
        run_checked(
            self._runner, "rclone deletefile",
            ["rclone", "deletefile", self._path(name)],
        )


class ClientCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    notes: int
    passages: int


@dataclass(frozen=True)
class DbFacts:
    """What a restored database must still say."""

    alembic_revision: str
    pgvector_version: str
    clients: Mapping[str, ClientCounts]


def read_db_facts(connection: Connection) -> DbFacts:
    """Read the facts on an open connection, so a caller holding a snapshot gets
    the snapshot's facts. The same query checks the restored copy."""
    revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    pgvector = connection.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one()
    notes = dict(
        connection.execute(
            text("SELECT client_name, count(*) FROM notes GROUP BY client_name")
        ).all()
    )
    passages = dict(
        connection.execute(
            text("SELECT client_name, count(*) FROM passages GROUP BY client_name")
        ).all()
    )
    return DbFacts(
        alembic_revision=revision,
        pgvector_version=pgvector,
        clients={
            client: ClientCounts(notes=notes.get(client, 0), passages=passages.get(client, 0))
            for client in sorted(notes.keys() | passages.keys())
        },
    )


class Manifest(BaseModel):
    """The sidecar. It holds client names and counts, never note content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dump: str
    created_at: datetime
    sha256: str
    size_bytes: int
    alembic_revision: str
    pgvector_version: str
    clients: dict[str, ClientCounts]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> Manifest:
        try:
            return cls.model_validate_json(raw)
        except ValidationError:
            # Field names only would still be noise; the point is that it failed.
            raise BackupError("the backup's manifest is malformed") from None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def describe_failure(exc: Exception) -> str:
    """A one-line account of a failure, safe for the journal.

    Only BackupError and the settings errors word their own messages. A database
    error is reduced to its class and SQLSTATE and an OS error to its errno,
    since their text can quote statements and file paths.
    """
    if isinstance(exc, BackupError):
        return str(exc)
    if isinstance(exc, SQLAlchemyError):
        orig = getattr(exc, "orig", None)
        sqlstate = getattr(orig, "sqlstate", None) or "unknown"
        return f"database error {type(orig or exc).__name__} (sqlstate {sqlstate})"
    if isinstance(exc, OSError):
        return f"{type(exc).__name__} (errno {exc.errno})"
    if isinstance(exc, ValidationError):
        # app.config's settings, which unlike BackupSettings do not hide their
        # input: a ValidationError's text could print DATABASE_URL.
        return "invalid settings in app.config; the API would refuse to start with them too"
    return type(exc).__name__
