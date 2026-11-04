"""scripts/backup.py against a real database and a real pg_dump in the container.

rclone is the stand-in from tests/_backup_helpers.py. A restore of what these
backups produce is in tests/test_restore_check.py.
"""
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from scripts import backup
from scripts._backup_common import (
    BackupError,
    Manifest,
    ToolError,
    ToolResult,
    backup_stem,
    libpq_target,
    parse_backup_name,
    run_tool,
)
from tests._backup_helpers import (
    ALEMBIC_INI,
    HEARTBEAT_URL,
    REMOTE,
    SECRET_BODY,
    SEEDED_CLIENTS,
    ContainerTools,
    FakeRclone,
    ToolRouter,
    make_settings,
    source_url,  # noqa: F401  (fixture)
)
from tests.factories import make_note

NOW = datetime(2026, 11, 2, 3, 30, tzinfo=UTC)
DUMP = f"{backup_stem(NOW)}.dump"
MANIFEST = f"{backup_stem(NOW)}.manifest.json"


@pytest.fixture
def rclone():
    return FakeRclone()


@pytest.fixture
def pg_tools(pg_container):
    return ContainerTools(pg_container)


@pytest.fixture
def router(rclone, pg_tools):
    return ToolRouter(rclone, pg_tools)


@pytest.fixture
def settings(tmp_path, source_url, pg_container):  # noqa: F811
    return make_settings(tmp_path, source_url, pg_container.get_connection_url())


def test_the_manifest_describes_the_dump_and_the_database(settings, router, rclone, tmp_path):
    backup.run_backup(settings, runner=router, now=NOW)

    assert set(rclone.files) == {DUMP, MANIFEST}
    dump = rclone.files[DUMP]
    assert dump.startswith(b"PGDMP"), "not a custom-format archive"

    raw = rclone.files[MANIFEST].decode("utf-8")
    manifest = Manifest.from_json(raw)
    assert manifest.dump == DUMP
    assert manifest.created_at == NOW
    assert manifest.sha256 == hashlib.sha256(dump).hexdigest()
    assert manifest.size_bytes == len(dump)
    head = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_current_head()
    assert manifest.alembic_revision == head
    assert manifest.pgvector_version
    assert json.loads(raw)["clients"] == SEEDED_CLIENTS
    assert SECRET_BODY not in raw

    # The spool holds the same two files and nothing half-written.
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([DUMP, MANIFEST])
    assert (tmp_path / DUMP).read_bytes() == dump


def test_the_manifest_arrives_after_the_dump(settings, router, rclone):
    backup.run_backup(settings, runner=router, now=NOW)
    uploads = [call[3] for call in rclone.calls if call[1] == "copyto"]
    assert uploads == [f"{REMOTE}/{DUMP}", f"{REMOTE}/{MANIFEST}"]


def test_a_write_during_the_dump_cannot_make_the_manifest_disagree(
    settings, rclone, pg_tools, source_url  # noqa: F811
):
    # The race the shared snapshot exists for: a note committed after the facts are
    # read and before pg_dump starts must be in neither, or a good backup would
    # fail its restore check. The dump's own contents are checked by restoring it.
    source = create_engine(source_url)
    router = ToolRouter(rclone, pg_tools)
    wrote = []

    def write_then_run(args, env=None):
        if args[0] == "pg_dump":
            with Session(source) as db:
                make_note(db, "acme", path_or_uri="late.md", body="late")
                db.commit()
            wrote.append(True)
        return router(args, env)

    try:
        report = backup.run_backup(settings, runner=write_then_run, now=NOW)
        assert wrote
        assert report.manifest.clients["acme"].notes == SEEDED_CLIENTS["acme"]["notes"]

        # The dump holds the snapshot too, not just the manifest.
        container_path = pg_tools.push(settings.backup_spool_dir / DUMP)
        _, listing = pg_tools.run(["pg_restore", "--data-only", "-f", "-", container_path])
        assert b"late.md" not in listing
    finally:
        with source.begin() as connection:
            connection.execute(text("DELETE FROM notes WHERE path_or_uri = 'late.md'"))
        source.dispose()


def test_a_remote_that_is_not_crypt_is_refused_before_the_dump(settings, pg_tools):
    rclone = FakeRclone(remote_type="s3")
    with pytest.raises(BackupError, match="must be an rclone crypt remote"):
        backup.run_backup(settings, runner=ToolRouter(rclone, pg_tools), now=NOW)
    assert rclone.commands() == ["listremotes"]
    assert list(settings.backup_spool_dir.iterdir()) == []


def test_a_remote_rclone_does_not_know_is_refused(settings, rclone, router):
    named_elsewhere = settings.model_copy(update={"backup_rclone_remote": "absent:nescio"})
    with pytest.raises(BackupError, match="does not know"):
        backup.run_backup(named_elsewhere, runner=router, now=NOW)


def test_a_failed_dump_leaves_nothing_and_uploads_nothing(settings, router, rclone, pg_tools):
    pg_tools.exits["pg_dump"] = 1
    (settings.backup_spool_dir / "nescio-20000101T000000Z.dump.partial").write_text("killed run")

    with pytest.raises(ToolError, match="pg_dump failed with exit code 1"):
        backup.run_backup(settings, runner=router, now=NOW)

    assert list(settings.backup_spool_dir.iterdir()) == []
    assert rclone.files == {}


def test_any_nonzero_rclone_exit_fails_the_run(settings, router, rclone):
    rclone.fail.add("copyto")
    with pytest.raises(ToolError, match="upload"):
        backup.run_backup(settings, runner=router, now=NOW)


def test_a_failing_upload_still_bounds_the_spool(settings, router, rclone):
    # Night after night of failed uploads must not fill the VPS's disk.
    spool = settings.backup_spool_dir
    old = NOW - timedelta(days=1)
    for name in (f"{backup_stem(old)}.dump", f"{backup_stem(old)}.manifest.json"):
        (spool / name).write_text("x")
    rclone.fail.add("copyto")
    keep_one = settings.model_copy(update={"backup_retention_daily": 1, "backup_retention_weekly": 0})

    with pytest.raises(ToolError):
        backup.run_backup(keep_one, runner=router, now=NOW)

    assert sorted(path.name for path in spool.iterdir()) == sorted([DUMP, MANIFEST])


def test_retention_prunes_the_remote_and_the_spool_after_a_good_run(settings, router, rclone):
    spool = settings.backup_spool_dir
    older = [NOW - timedelta(days=age) for age in (1, 2, 3)]
    old_names = [
        f"{backup_stem(moment)}{suffix}"
        for moment in older
        for suffix in (".dump", ".manifest.json")
    ]
    for name in old_names:
        rclone.files[name] = b"x"
        (spool / name).write_text("x")
    rclone.files["someone-elses-file.txt"] = b"x"

    report = backup.run_backup(
        settings.model_copy(update={"backup_retention_daily": 2, "backup_retention_weekly": 0}),
        runner=router,
        now=NOW,
    )

    kept = [f"{backup_stem(older[0])}{suffix}" for suffix in (".dump", ".manifest.json")]
    assert sorted(rclone.files) == sorted([DUMP, MANIFEST, *kept, "someone-elses-file.txt"])
    assert sorted(path.name for path in spool.iterdir()) == sorted([DUMP, MANIFEST, *kept])
    assert report.pruned_remote == report.pruned_spool == 4


def test_the_password_stays_out_of_every_argv(settings, rclone, pg_tools):
    seen = []
    router = ToolRouter(rclone, pg_tools)

    def recording(args, env=None):
        seen.append((list(args), dict(env or {})))
        return router(args, env)

    backup.run_backup(settings, runner=recording, now=NOW)

    pg_dump_args, pg_dump_env = next(call for call in seen if call[0][0] == "pg_dump")
    assert pg_dump_env["PGPASSWORD"] == "test"
    assert not any(":test@" in arg for args, _ in seen for arg in args)
    assert "--no-owner" in pg_dump_args
    assert "-Fc" in pg_dump_args


def test_libpq_target_strips_the_driver_and_keeps_the_query():
    target = libpq_target("postgresql+psycopg://u:p%40ss@db:5433/x?sslmode=require")
    assert target.uri == "postgresql://u@db:5433/x?sslmode=require"
    assert target.env["PGPASSWORD"] == "p@ss"


def test_run_tool_keeps_stderr_to_itself(capfd):
    result = run_tool(
        [sys.executable, "-c", "import sys; sys.stderr.write('note content'); sys.exit(3)"]
    )
    assert result.returncode == 3
    captured = capfd.readouterr()
    assert "note content" not in captured.out + captured.err


def test_run_tool_reports_a_missing_binary_as_127():
    assert run_tool(["nescio-no-such-tool"]).returncode == 127


def _env(monkeypatch, tmp_path, source, restore):
    # chdir: no developer's .env reaches BackupSettings.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", source)
    monkeypatch.setenv("RESTORE_CHECK_DATABASE_URL", restore)
    monkeypatch.setenv("BACKUP_SPOOL_DIR", str(tmp_path))
    monkeypatch.setenv("BACKUP_RCLONE_REMOTE", REMOTE)
    monkeypatch.setenv("BACKUP_HEARTBEAT_URL", HEARTBEAT_URL)


def test_main_reports_the_backup_without_a_secret_or_a_path(
    monkeypatch, tmp_path, capsys, router, rclone, source_url, pg_container  # noqa: F811
):
    _env(monkeypatch, tmp_path, source_url, pg_container.get_connection_url())
    capsys.readouterr()

    assert backup.main([], runner=router) == 0

    captured = capsys.readouterr()
    assert "backup uploaded" in captured.out
    assert "passages=7" in captured.out
    assert parse_backup_name(captured.out.split("dump=")[1].split()[0])
    for secret in (source_url, HEARTBEAT_URL, SECRET_BODY, str(tmp_path)):
        assert secret not in captured.out + captured.err


def test_main_exits_1_on_failure(monkeypatch, tmp_path, capsys, router, pg_tools, source_url, pg_container):  # noqa: F811
    _env(monkeypatch, tmp_path, source_url, pg_container.get_connection_url())
    pg_tools.exits["pg_dump"] = 7
    capsys.readouterr()

    assert backup.main([], runner=router) == 1

    assert "pg_dump failed with exit code 7" in capsys.readouterr().err


def test_main_exits_2_on_bad_settings_before_running_a_tool(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p4ssw0rd@db/x")
    monkeypatch.setenv("RESTORE_CHECK_DATABASE_URL", "postgresql://u:p4ssw0rd@db/y")
    monkeypatch.setenv("BACKUP_SPOOL_DIR", str(tmp_path))
    monkeypatch.setenv("BACKUP_RCLONE_REMOTE", "not-a-remote")
    capsys.readouterr()

    def no_tools(args, env=None) -> ToolResult:
        raise AssertionError("no tool may run on bad settings")

    assert backup.main([], runner=no_tools) == 2

    captured = capsys.readouterr()
    assert "BACKUP_RCLONE_REMOTE" in captured.err
    assert "p4ssw0rd" not in captured.out + captured.err
