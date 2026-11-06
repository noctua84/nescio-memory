"""The restore check, performed: real pg_dump, real pg_restore, real database.

This is the CI exercise #75 asks for. Every run of the suite takes a backup with
scripts/backup.py, restores it with scripts/restore_check.py inside the
testcontainers Postgres, and verifies what came back. Only rclone and the
heartbeat's HTTP transport are stubs. tests/test_restore_verification.py covers
each verification function on its own.
"""
import hashlib
import json

import httpx
import pytest
from sqlalchemy import text

from scripts import restore_check, restore_verification
from scripts._backup_common import BackupError, Manifest, make_engine
from scripts.restore_check import (
    HeartbeatError,
    Refused,
    assert_server_can_restore,
    create_scratch_database,
    drop_scratch_database,
    ping_heartbeat,
    restore_dump,
    validate_scratch_db_name,
)
from scripts.restore_verification import VerificationError
from tests._backup_helpers import (
    HEARTBEAT_URL,
    REMOTE,
    SECRET_BODY,
    ContainerTools,
    FakeRclone,
    ToolRouter,
    admin_engine,
    backed_up,  # noqa: F401  (fixture)
    scratch_databases,
    source_url,  # noqa: F401  (fixture)
)


class Heartbeat:
    """A stub transport standing in for the dead-man's switch."""

    def __init__(self, *responses: int):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.responses.pop(0) if self.responses else 200
        return httpx.Response(status)


@pytest.fixture
def rclone(backed_up):  # noqa: F811
    return FakeRclone(backed_up)


@pytest.fixture
def pg_tools(pg_container):
    return ContainerTools(pg_container)


@pytest.fixture
def router(rclone, pg_tools):
    return ToolRouter(rclone, pg_tools)


@pytest.fixture
def run_main(monkeypatch, tmp_path, capsys, router, source_url, pg_container):  # noqa: F811
    """main() as the timer runs it. Returns (exit code, captured, heartbeat)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", source_url)
    monkeypatch.setenv("RESTORE_CHECK_DATABASE_URL", pg_container.get_connection_url())
    monkeypatch.setenv("BACKUP_SPOOL_DIR", str(tmp_path))
    monkeypatch.setenv("BACKUP_RCLONE_REMOTE", REMOTE)
    monkeypatch.setenv("BACKUP_HEARTBEAT_URL", HEARTBEAT_URL)
    capsys.readouterr()

    def run(*argv, heartbeat=None, runner=None):
        heartbeat = heartbeat or Heartbeat()
        code = restore_check.main(
            ["--scheduled", *argv],
            runner=runner or router,
            transport=heartbeat.transport,
            sleep=lambda seconds: None,
        )
        return code, capsys.readouterr(), heartbeat

    return run


def _manifest_name(rclone: FakeRclone) -> str:
    return next(name for name in rclone.files if name.endswith(".manifest.json"))


def _dump_name(rclone: FakeRclone) -> str:
    return next(name for name in rclone.files if name.endswith(".dump"))


def _tamper(rclone: FakeRclone, **changes) -> None:
    name = _manifest_name(rclone)
    data = json.loads(rclone.files[name])
    data.update(changes)
    rclone.files[name] = json.dumps(data).encode("utf-8")


def test_a_good_backup_verifies_pings_once_and_cleans_up(run_main, pg_container, tmp_path):
    code, captured, heartbeat = run_main()

    assert code == 0, captured.err
    assert "restore verified heartbeat=sent" in captured.out
    assert "pg_restore_s=" in captured.out
    assert [(r.method, str(r.url)) for r in heartbeat.requests] == [("GET", HEARTBEAT_URL)]
    assert scratch_databases(pg_container) == []
    assert list(tmp_path.iterdir()) == []


def _count_off_by_one(rclone, pg_tools):
    data = json.loads(rclone.files[_manifest_name(rclone)])
    clients = data["clients"]
    clients["acme"]["notes"] += 1
    _tamper(rclone, clients=clients)


def _client_missing(rclone, pg_tools):
    data = json.loads(rclone.files[_manifest_name(rclone)])
    del data["clients"]["globex"]
    _tamper(rclone, clients=data["clients"])


def _client_invented(rclone, pg_tools):
    data = json.loads(rclone.files[_manifest_name(rclone)])
    data["clients"]["initech"] = {"notes": 1, "passages": 1}
    _tamper(rclone, clients=data["clients"])


def _dump_replaced_by_garbage_with_a_matching_sha(rclone, pg_tools):
    garbage = b"this is not an archive"
    rclone.files[_dump_name(rclone)] = garbage
    _tamper(rclone, sha256=hashlib.sha256(garbage).hexdigest())


FAILURES = {
    "count off by one": _count_off_by_one,
    "client missing from manifest": _client_missing,
    "client invented in manifest": _client_invented,
    "wrong alembic revision": lambda r, p: _tamper(r, alembic_revision="deadbeef"),
    "sha256 mismatch": lambda r, p: _tamper(r, sha256="0" * 64),
    "manifest for another dump": lambda r, p: _tamper(r, dump="nescio-20000101T000000Z.dump"),
    "malformed manifest": lambda r, p: r.files.update({_manifest_name(r): b"{not json"}),
    "missing manifest": lambda r, p: r.files.pop(_manifest_name(r)),
    "empty remote": lambda r, p: r.files.clear(),
    "rclone listing fails": lambda r, p: r.fail.add("lsjson"),
    "rclone download fails": lambda r, p: r.fail.add("copyto"),
    "pg_restore exits non-zero": lambda r, p: p.exits.update({"pg_restore": 1}),
    "corrupt archive": _dump_replaced_by_garbage_with_a_matching_sha,
}


@pytest.mark.parametrize("failure", FAILURES.values(), ids=FAILURES.keys())
def test_no_failure_pings_the_heartbeat(failure, run_main, rclone, pg_tools, pg_container, tmp_path):
    failure(rclone, pg_tools)

    code, captured, heartbeat = run_main()

    assert code == 1
    assert "restore check failed" in captured.err
    assert "restore verified" not in captured.out
    assert heartbeat.requests == []
    assert scratch_databases(pg_container) == []
    assert list(tmp_path.iterdir()) == []


def test_an_older_complete_backup_does_not_stand_in_for_a_newer_broken_one(run_main, rclone):
    # Falling back would keep the heartbeat alive while nothing new reached the remote.
    rclone.files["nescio-29990101T000000Z.dump"] = b"upload died before the manifest"

    code, captured, heartbeat = run_main()

    assert code == 1
    assert "has no manifest" in captured.err
    assert heartbeat.requests == []


@pytest.mark.parametrize(
    "assertion",
    [
        "assert_alembic_revision",
        "assert_client_counts",
        "assert_hnsw_index_valid",
        "assert_search_finds_a_stored_passage",
    ],
)
def test_each_assertion_is_part_of_the_check_and_blocks_the_ping(
    assertion, run_main, monkeypatch, pg_container
):
    def failing(*args):
        raise VerificationError("stubbed failure")

    monkeypatch.setattr(restore_verification, assertion, failing)

    code, captured, heartbeat = run_main()

    assert code == 1
    assert "stubbed failure" in captured.err
    assert heartbeat.requests == []
    assert scratch_databases(pg_container) == []


def test_a_scratch_database_that_cannot_be_dropped_blocks_the_ping(
    run_main, monkeypatch, pg_container
):
    real_drop = restore_check.drop_scratch_database

    def failing(admin, name):
        raise BackupError("stubbed drop failure")

    monkeypatch.setattr(restore_check, "drop_scratch_database", failing)
    try:
        code, _, heartbeat = run_main()
    finally:
        engine = make_engine(pg_container.get_connection_url())
        real_drop(engine, restore_check.DEFAULT_SCRATCH_DB)
        engine.dispose()

    assert code == 1
    assert heartbeat.requests == []


def test_a_leftover_scratch_database_is_replaced(run_main, pg_container):
    admin = admin_engine(pg_container)
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{restore_check.DEFAULT_SCRATCH_DB}"'))
    admin.dispose()

    code, captured, _ = run_main()

    assert code == 0, captured.err
    assert scratch_databases(pg_container) == []


def test_a_heartbeat_that_never_lands_fails_the_run(run_main, pg_container):
    heartbeat = Heartbeat(500, 500, 500)

    code, captured, _ = run_main(heartbeat=heartbeat)

    assert code == 1
    assert "does not count as verified" in captured.err
    assert "restore verified" not in captured.out
    assert len(heartbeat.requests) == 3


def test_the_logs_hold_no_secret_and_no_path(run_main, source_url, pg_container, tmp_path):  # noqa: F811
    for outcome in (run_main(), run_main("--scratch-db", "BAD")):
        _, captured, _ = outcome
        output = captured.out + captured.err
        for secret in (
            HEARTBEAT_URL,
            SECRET_BODY,
            source_url,
            pg_container.get_connection_url(),
            str(tmp_path),
        ):
            assert secret not in output


def _no_tools(args, env=None):
    raise AssertionError("a refused run must not start any tool")


def test_scheduled_mode_refuses_to_run_without_a_heartbeat_url(run_main, monkeypatch):
    monkeypatch.delenv("BACKUP_HEARTBEAT_URL")

    code, captured, heartbeat = run_main(runner=_no_tools)

    assert code == 2
    assert "BACKUP_HEARTBEAT_URL" in captured.err
    assert heartbeat.requests == []


def test_a_manual_run_without_a_heartbeat_url_verifies_and_says_so(
    run_main, monkeypatch, capsys, router
):
    monkeypatch.delenv("BACKUP_HEARTBEAT_URL")

    code = restore_check.main([], runner=router)

    assert code == 0
    assert "heartbeat=not_configured" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "nescio_memory",
        "postgres",
        "restore_check_",
        "RESTORE_CHECK_x",
        "xrestore_check_x",
        'restore_check_x"; DROP DATABASE nescio_memory; --',
        "restore_check_" + "x" * 50,
    ],
)
def test_the_scratch_name_guard_holds_everywhere_a_name_is_used(name, run_main):
    with pytest.raises(Refused):
        validate_scratch_db_name(name)
    # None for the engine: reaching it would raise something else.
    with pytest.raises(Refused):
        create_scratch_database(None, name)
    with pytest.raises(Refused):
        drop_scratch_database(None, name)
    with pytest.raises(Refused):
        restore_dump(_no_tools, "any.dump", "postgresql://u@h/d", name)

    code, captured, heartbeat = run_main("--scratch-db", name, runner=_no_tools)
    assert code == 2
    assert "refusing scratch database name" in captured.err
    assert heartbeat.requests == []


@pytest.mark.parametrize("name", ["restore_check_scratch", "restore_check_20261102", "restore_check_a"])
def test_the_scratch_name_guard_accepts_its_own_prefix(name):
    assert validate_scratch_db_name(name) == name


def _manifest(pgvector_version: str) -> Manifest:
    return Manifest(
        dump="d", created_at="2026-11-02T00:00:00Z", sha256="s", size_bytes=0,
        alembic_revision="r", pgvector_version=pgvector_version, clients={},
    )


def test_a_role_that_cannot_create_the_extension_is_named_as_the_cause(pg_container):
    role = "restore_check_plain_role"
    admin = admin_engine(pg_container)
    with admin.connect() as connection:
        connection.execute(text(f"DROP ROLE IF EXISTS {role}"))
        connection.execute(text(f"CREATE ROLE {role} LOGIN CREATEDB PASSWORD 'x'"))
    url = pg_container.get_connection_url().replace("test:test@", f"{role}:x@")
    engine = make_engine(url)
    manifest = _manifest("0.1.0")
    try:
        with pytest.raises(VerificationError, match="must be a superuser"):
            assert_server_can_restore(engine, manifest)
    finally:
        engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f"DROP ROLE {role}"))
        admin.dispose()


def test_a_server_with_an_older_pgvector_is_named_as_the_cause(pg_container):
    engine = make_engine(pg_container.get_connection_url())
    manifest = _manifest("99.0.0")
    try:
        with pytest.raises(VerificationError, match="older than"):
            assert_server_can_restore(engine, manifest)
    finally:
        engine.dispose()


def test_the_ping_retries_a_transient_failure():
    heartbeat, sleeps = Heartbeat(503, 200), []
    ping_heartbeat(HEARTBEAT_URL, transport=heartbeat.transport, sleep=sleeps.append)
    assert len(heartbeat.requests) == 2
    assert len(sleeps) == 1


def test_the_ping_gives_up_without_naming_the_url():
    attempts = []

    def refuse(request):
        attempts.append(request)
        raise httpx.ConnectError(f"cannot reach {request.url}")

    transport = httpx.MockTransport(refuse)
    with pytest.raises(HeartbeatError) as caught:
        ping_heartbeat(HEARTBEAT_URL, transport=transport, sleep=lambda seconds: None)
    assert len(attempts) == 3
    assert HEARTBEAT_URL not in str(caught.value)
    assert "ConnectError" in str(caught.value)


def test_the_ping_never_retries_past_its_time_bound():
    # An attempt that uses its whole 30 s worst case leaves no room for another
    # 30 s one inside the minute, so none is started.
    now = [0.0]
    attempts = []

    def slow(request):
        attempts.append(request)
        now[0] += 30.0
        return httpx.Response(500)

    transport = httpx.MockTransport(slow)
    with pytest.raises(HeartbeatError):
        ping_heartbeat(
            HEARTBEAT_URL,
            transport=transport,
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            clock=lambda: now[0],
        )
    assert len(attempts) == 1


def test_a_redirect_is_not_a_ping():
    heartbeat = Heartbeat(302, 302, 302)
    with pytest.raises(HeartbeatError, match="HTTP 302"):
        ping_heartbeat(HEARTBEAT_URL, transport=heartbeat.transport, sleep=lambda seconds: None)
