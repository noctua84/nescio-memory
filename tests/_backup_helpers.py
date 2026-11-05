"""The rig for the backup and restore-check tests.

pg_dump and pg_restore are real and run inside the testcontainers Postgres, whose
image ships both, so these tests need neither on the host. rclone is a stand-in:
tests must not need it installed, and FakeRclone answers the same four commands
the scripts use. Files cross between the host spool and the container through the
Docker API, which is what lets scripts/backup.py and scripts/restore_check.py run
whole and unchanged.

Backups can only see committed data, so `source_url` is a database of its own,
migrated and seeded for the module and dropped afterwards. The `connection`
fixture's rolled-back transaction would be invisible to pg_dump, and committing
to the shared database would leak rows into every other test.
"""
import io
import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from alembic import command
from app.config import settings as app_settings
from scripts import backup
from scripts._backup_common import ToolResult
from scripts.backup_settings import BackupSettings
from tests.factories import make_note, make_passage
from tests.fakes import fake_embedding

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"
REMOTE = "secret:nescio"
HEARTBEAT_URL = "https://hc-ping.example/5f4dcc3b-heartbeat-secret"
SECRET_BODY = "NOTE-BODY-THAT-MUST-NEVER-REACH-A-LOG"
SOURCE_DB = "backup_source"

# What seed_source writes, which the manifest must report.
SEEDED_CLIENTS = {
    "acme": {"notes": 3, "passages": 4},
    "globex": {"notes": 1, "passages": 3},
}


class FakeRclone:
    """An in-memory remote that answers exactly what the scripts ask of rclone."""

    def __init__(self, files: Mapping[str, bytes] | None = None, remote_type: str = "crypt"):
        self.files = dict(files or {})
        self.remote_type = remote_type
        self.calls: list[list[str]] = []
        self.fail: set[str] = set()

    def __call__(self, args: Sequence[str], env: Mapping[str, str] | None = None) -> ToolResult:
        self.calls.append(list(args))
        command_name = args[1]
        if command_name in self.fail:
            return ToolResult(1, "")
        if command_name == "listremotes":
            return ToolResult(0, f"secret: {self.remote_type}\nother: s3\n")
        if command_name == "lsjson":
            return ToolResult(0, json.dumps([{"Name": name, "Path": name} for name in self.files]))
        if command_name == "copyto":
            source, destination = args[2], args[3]
            if destination.startswith(f"{REMOTE}/"):
                self.files[destination.removeprefix(f"{REMOTE}/")] = Path(source).read_bytes()
            else:
                Path(destination).write_bytes(self.files[source.removeprefix(f"{REMOTE}/")])
            return ToolResult(0, "")
        if command_name == "deletefile":
            del self.files[args[2].removeprefix(f"{REMOTE}/")]
            return ToolResult(0, "")
        raise AssertionError(f"the scripts should not run rclone {command_name}")

    def commands(self) -> list[str]:
        return [call[1] for call in self.calls]


class ContainerTools:
    """Runs pg_dump and pg_restore inside the Postgres container.

    The scripts name host paths and a host-mapped port. This maps them: --dbname
    to the server as seen from inside, a --file output back out to the host, and a
    dump argument in. `exits` lets a test make a tool fail.
    """

    CONTAINER_DIR = "/tmp"

    def __init__(self, pg_container):
        self._container = pg_container.get_wrapped_container()
        self.exits: dict[str, int] = {}

    def push(self, host_path: Path) -> str:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            archive.add(host_path, arcname=host_path.name)
        self._container.put_archive(self.CONTAINER_DIR, buffer.getvalue())
        return f"{self.CONTAINER_DIR}/{host_path.name}"

    def _pull(self, container_path: str, host_path: Path) -> None:
        stream, _ = self._container.get_archive(container_path)
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as archive:
            member = archive.extractfile(archive.getmembers()[0])
            host_path.write_bytes(member.read())

    def run(self, args: Sequence[str], env: Mapping[str, str] | None = None) -> tuple[int, bytes]:
        return self._container.exec_run(list(args), environment=dict(env or {}))

    def __call__(self, args: Sequence[str], env: Mapping[str, str] | None = None) -> ToolResult:
        if args[0] in self.exits:
            return ToolResult(self.exits[args[0]], "")
        mapped, pull = [], None
        for arg in args:
            if arg.startswith("--dbname="):
                url = make_url(arg.removeprefix("--dbname=")).set(host="localhost", port=5432)
                arg = f"--dbname={url.render_as_string(hide_password=False)}"
            elif arg.startswith("--file="):
                host_path = Path(arg.removeprefix("--file="))
                pull = (f"{self.CONTAINER_DIR}/{host_path.name}", host_path)
                arg = f"--file={pull[0]}"
            elif arg.endswith(".dump"):
                arg = self.push(Path(arg))
            mapped.append(arg)
        exit_code, _output = self.run(mapped, env)
        if exit_code == 0 and pull is not None:
            self._pull(*pull)
        return ToolResult(exit_code, "")


class ToolRouter:
    """The runner the scripts get: rclone to the fake, the pg tools to the container."""

    def __init__(self, rclone: FakeRclone, pg: ContainerTools):
        self.rclone = rclone
        self.pg = pg

    def __call__(self, args: Sequence[str], env: Mapping[str, str] | None = None) -> ToolResult:
        return (self.rclone if args[0] == "rclone" else self.pg)(args, env)


def seed_source(db: Session) -> None:
    """Enough shape to fail on: two clients, an embedding shared by two notes, and
    a degenerate zero vector.

    The shared embedding must never count as a search miss, whichever twin comes back.
    """
    twin_a = make_note(db, "acme", path_or_uri="twin_a.md", body=SECRET_BODY)
    twin_b = make_note(db, "acme", path_or_uri="twin_b.md", body=SECRET_BODY)
    make_passage(db, twin_a, ordinal=0, content="the same text", embedding=fake_embedding("twin"))
    make_passage(db, twin_b, ordinal=0, content="the same text", embedding=fake_embedding("twin"))
    plain = make_note(db, "acme", path_or_uri="plain.md", body=SECRET_BODY)
    make_passage(db, plain, ordinal=0, content="first plain passage")
    make_passage(db, plain, ordinal=1, content="second plain passage")

    other = make_note(db, "globex", path_or_uri="other.md", body=SECRET_BODY)
    make_passage(db, other, ordinal=0, content="another tenant's passage")
    make_passage(db, other, ordinal=1, content="and one more of theirs")
    make_passage(
        db, other, ordinal=2, content="zero vector",
        embedding=[0.0] * app_settings.embedding_dimension,
    )
    db.commit()


def database_url(pg_container, database: str) -> str:
    url = make_url(pg_container.get_connection_url()).set(database=database)
    return url.render_as_string(hide_password=False)


def admin_engine(pg_container):
    return create_engine(pg_container.get_connection_url(), isolation_level="AUTOCOMMIT")


@pytest.fixture(scope="module")
def source_url(pg_container):
    """A committed, migrated, seeded database nothing else in the suite touches."""
    admin = admin_engine(pg_container)
    with admin.connect() as connection:
        connection.execute(text(f"DROP DATABASE IF EXISTS {SOURCE_DB}"))
        connection.execute(text(f"CREATE DATABASE {SOURCE_DB}"))

    url = database_url(pg_container, SOURCE_DB)
    source = create_engine(url)
    alembic_config = Config(str(ALEMBIC_INI))
    with source.begin() as connection:
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")
    with Session(source) as db:
        seed_source(db)
    source.dispose()

    yield url

    with admin.connect() as connection:
        connection.execute(text(f"DROP DATABASE IF EXISTS {SOURCE_DB} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture(scope="module")
def backed_up(source_url, pg_container, tmp_path_factory):
    """The remote's files after one real backup. Tests copy them to tamper freely."""
    rclone = FakeRclone()
    settings = make_settings(
        tmp_path_factory.mktemp("spool"), source_url, pg_container.get_connection_url()
    )
    backup.run_backup(settings, runner=ToolRouter(rclone, ContainerTools(pg_container)))
    return rclone.files


def make_settings(
    spool: Path, source: str, restore: str, heartbeat: str | None = HEARTBEAT_URL
) -> BackupSettings:
    return BackupSettings(
        _env_file=None,
        database_url=source,
        restore_check_database_url=restore,
        backup_spool_dir=spool,
        backup_rclone_remote=REMOTE,
        backup_heartbeat_url=heartbeat,
    )


def scratch_databases(pg_container) -> list[str]:
    admin = admin_engine(pg_container)
    with admin.connect() as connection:
        names = connection.execute(
            text("SELECT datname FROM pg_database WHERE datname LIKE 'restore\\_check\\_%'")
        ).scalars().all()
    admin.dispose()
    return list(names)
