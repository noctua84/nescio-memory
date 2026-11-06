"""Dump the database, write its manifest, upload both, prune old backups.

    uv run python -m scripts.backup

Invoked with -m, not as a path: `python scripts/backup.py` puts scripts/ on
sys.path instead of the repository root. Configuration is the environment or
.env; see the "Backup and recovery" section of the README. Needs pg_dump and
rclone on PATH, and the configured remote must be an rclone crypt remote.

Exit codes: 0 backup uploaded and pruned, 1 failed, 2 invalid settings.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from scripts._backup_common import (
    DUMP_SUFFIX,
    MANIFEST_SUFFIX,
    BackupError,
    DbFacts,
    LibpqTarget,
    Manifest,
    RcloneRemote,
    ToolRunner,
    backup_stem,
    describe_failure,
    libpq_target,
    make_engine,
    parse_backup_name,
    read_db_facts,
    run_checked,
    run_tool,
    sha256_file,
)
from scripts.backup_settings import BackupSettings, describe_settings_error

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE_ERROR = 2

_PARTIAL_SUFFIX = ".partial"


def select_retained(
    moments: Iterable[datetime], daily: int, weekly: int
) -> set[datetime]:
    """The backups to keep: the newest of each of the `daily` most recent days
    that have one, and of each of the `weekly` most recent ISO weeks that have one.

    Pure, and it counts backups rather than the calendar. It takes no clock on
    purpose: when backups stop, a calendar window would age the last good ones
    out, and the worst moment to lose them is after a long silent outage. Here
    the newest backup is always kept, whatever its age, since daily is at least 1.
    The two windows overlap: a kept weekly can also be a kept daily.
    """
    ordered = sorted(set(moments), reverse=True)
    retained: set[datetime] = set()
    for window, bucket in (
        (daily, lambda moment: moment.date()),
        (weekly, lambda moment: moment.isocalendar()[:2]),
    ):
        seen = set()
        for moment in ordered:
            key = bucket(moment)
            if key in seen:
                continue
            if len(seen) == window:
                break
            seen.add(key)
            retained.add(moment)
    return retained


def plan_prune(names: Iterable[str], daily: int, weekly: int) -> list[str]:
    """Which of these file names to delete. Pure, like select_retained.

    Names this tool did not write are never returned. Only a complete backup, a
    dump with its manifest, counts towards retention: half of one left by a failed
    upload cannot be verified, so it takes no slot and is deleted.
    """
    parsed = {name: backup for name in names if (backup := parse_backup_name(name))}
    dumps = {moment for moment, suffix in parsed.values() if suffix == DUMP_SUFFIX}
    manifests = {moment for moment, suffix in parsed.values() if suffix == MANIFEST_SUFFIX}
    retained = select_retained(dumps & manifests, daily, weekly)
    return sorted(name for name, (moment, _) in parsed.items() if moment not in retained)


def dump_with_facts(
    engine: Engine, target: LibpqTarget, dump_path: Path | str, runner: ToolRunner
) -> DbFacts:
    """pg_dump the database, and read the manifest's facts from the same snapshot.

    Counted separately, an ingest landing between the count and the dump would
    make a good backup fail its restore check, and an alert that fires on healthy
    backups stops being read. So this transaction exports its snapshot, reads the
    facts, and pg_dump imports that snapshot; the transaction stays open until
    pg_dump exits because an exported snapshot lives only as long as its exporter.

    --no-owner is ignored for an archive and takes effect at pg_restore. It is
    passed here too so that the dump command says what the backup is.
    """
    connection = engine.connect().execution_options(
        isolation_level="REPEATABLE READ", postgresql_readonly=True
    )
    with connection, connection.begin():
        snapshot = connection.execute(text("SELECT pg_export_snapshot()")).scalar_one()
        facts = read_db_facts(connection)
        run_checked(
            runner,
            "pg_dump",
            [
                "pg_dump",
                "-Fc",
                "--no-owner",
                f"--snapshot={snapshot}",
                f"--file={dump_path}",
                f"--dbname={target.uri}",
            ],
            target.env,
        )
    return facts


def _create_spool_copy(
    settings: BackupSettings, moment: datetime, runner: ToolRunner
) -> Manifest:
    """Write the dump and its manifest into the spool under their final names.

    Both are written as .partial and renamed once complete, so a file with a
    final name is always whole. Leftovers of a killed run are swept first and
    this run's are swept on the way out: no part of a dump outlives its run.
    """
    spool = settings.backup_spool_dir
    stem = backup_stem(moment)
    dump_name = f"{stem}{DUMP_SUFFIX}"
    partial_dump = spool / f"{dump_name}{_PARTIAL_SUFFIX}"
    partial_manifest = spool / f"{stem}{MANIFEST_SUFFIX}{_PARTIAL_SUFFIX}"

    _sweep_partials(spool)
    try:
        engine = make_engine(settings.database_url)
        try:
            facts = dump_with_facts(
                engine, libpq_target(settings.database_url), partial_dump, runner
            )
        finally:
            engine.dispose()

        manifest = Manifest(
            dump=dump_name,
            created_at=moment,
            sha256=sha256_file(partial_dump),
            size_bytes=partial_dump.stat().st_size,
            alembic_revision=facts.alembic_revision,
            pgvector_version=facts.pgvector_version,
            clients=dict(facts.clients),
        )
        partial_manifest.write_text(manifest.to_json(), encoding="utf-8")
        os.replace(partial_dump, spool / dump_name)
        os.replace(partial_manifest, spool / f"{stem}{MANIFEST_SUFFIX}")
    finally:
        _sweep_partials(spool)
    return manifest


def _sweep_partials(spool: Path) -> None:
    for leftover in spool.glob(f"nescio-*{_PARTIAL_SUFFIX}"):
        leftover.unlink()


def prune_spool(spool: Path, daily: int, weekly: int) -> int:
    names = [path.name for path in spool.iterdir() if path.is_file()]
    doomed = plan_prune(names, daily, weekly)
    for name in doomed:
        (spool / name).unlink()
    return len(doomed)


def prune_remote(
    remote: RcloneRemote, uploaded: Sequence[str], daily: int, weekly: int
) -> int:
    """Delete what retention no longer keeps. A listing that does not show the
    files just uploaded is not a listing to delete on the strength of."""
    names = remote.list_names()
    if not set(uploaded) <= set(names):
        raise BackupError(
            "the remote listing does not show the backup just uploaded; "
            "refusing to prune"
        )
    doomed = plan_prune(names, daily, weekly)
    for name in doomed:
        remote.delete(name)
    return len(doomed)


@dataclass(frozen=True)
class BackupReport:
    manifest: Manifest
    pruned_spool: int
    pruned_remote: int


def run_backup(
    settings: BackupSettings,
    *,
    runner: ToolRunner = run_tool,
    now: datetime | None = None,
) -> BackupReport:
    daily = settings.backup_retention_daily
    weekly = settings.backup_retention_weekly
    remote = RcloneRemote(runner, settings.backup_rclone_remote)
    # Before the dump: finding out afterwards that the remote is not encrypted
    # costs a whole dump to say what this says in a second.
    remote.assert_crypt()

    moment = now or datetime.now(UTC).replace(microsecond=0)
    manifest = _create_spool_copy(settings, moment, runner)

    # The spool is pruned before the upload, not after: an upload that keeps
    # failing would otherwise add a full dump to the VPS's disk every night
    # until the disk is what fails.
    spool = settings.backup_spool_dir
    pruned_spool = prune_spool(spool, daily, weekly)

    # The manifest goes last, so its presence on the remote means the dump
    # arrived whole.
    manifest_name = f"{backup_stem(moment)}{MANIFEST_SUFFIX}"
    remote.upload(spool / manifest.dump, manifest.dump)
    remote.upload(spool / manifest_name, manifest_name)

    pruned_remote = prune_remote(remote, [manifest.dump, manifest_name], daily, weekly)
    return BackupReport(manifest, pruned_spool, pruned_remote)


def main(argv: Sequence[str] | None = None, runner: ToolRunner = run_tool) -> int:
    argparse.ArgumentParser(
        prog="python -m scripts.backup", description=__doc__.splitlines()[0]
    ).parse_args(sys.argv[1:] if argv is None else list(argv))

    try:
        settings = BackupSettings()
    except ValidationError as exc:
        print(f"error: {describe_settings_error(exc)}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    # Dumps hold personal notes: nothing this process writes is for anyone else.
    previous_umask = os.umask(0o077)
    try:
        report = run_backup(settings, runner=runner)
    except (BackupError, SQLAlchemyError, OSError) as exc:
        print(f"error: backup failed: {describe_failure(exc)}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        os.umask(previous_umask)

    manifest = report.manifest
    print(
        f"backup uploaded dump={manifest.dump} size_bytes={manifest.size_bytes} "
        f"clients={len(manifest.clients)} "
        f"notes={sum(counts.notes for counts in manifest.clients.values())} "
        f"passages={sum(counts.passages for counts in manifest.clients.values())} "
        f"pruned_spool={report.pruned_spool} pruned_remote={report.pruned_remote}"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
