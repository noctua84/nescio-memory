"""Settings for scripts/backup.py and scripts/restore_check.py.

Not part of app/config.py: backup.py must run without the API's settings, and
those settings cannot be imported without building them. The house style is the
same, though: every bound is a model validator whose error names the env var,
and a wrong value stops the run at startup rather than half way through a night.

Two values are secret or carry a secret: both database URLs hold a password and
the heartbeat URL is itself the credential of the dead-man's switch. So they are
left out of repr(), hide_input_in_errors keeps pydantic from printing the whole
input in a ValidationError, and no message below echoes a URL or a path. An
error reaches the journal; the secret must not ride along.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

# name:path. The name takes the characters rclone allows and at least two of
# them, so a Windows drive (C:\backups) is not read as a remote; the on-the-fly
# forms (:s3:bucket, name,opt=value:path) are refused because they can carry
# credentials on a command line and bypass the crypt check in the config file.
_REMOTE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-. +@]+:\S.*")


class BackupSettings(BaseSettings):
    # The source. Reused rather than renamed so the backup reads the same
    # database the API writes to, with no second URL to drift out of step.
    database_url: str = Field(repr=False)
    # Holds dumps of personal notes: the CronJob's pod security context creates
    # it 0700 (fsGroup plus the image's own directory mode).
    backup_spool_dir: Path = Path("/var/lib/nescio-backup")
    # Must name an rclone crypt remote. The form is checked here; the type is
    # checked against rclone's own config by scripts/_backup_common.py.
    backup_rclone_remote: str
    backup_retention_daily: int = 14
    backup_retention_weekly: int = 8
    # Optional here because a manual restore needs no dead-man's switch;
    # restore_check --scheduled refuses to run without it.
    backup_heartbeat_url: str | None = Field(default=None, repr=False)
    # A maintenance database on the server that hosts the scratch database.
    restore_check_database_url: str = Field(repr=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # .env is shared with the API, so its settings are expected here.
        extra="ignore",
        hide_input_in_errors=True,
    )

    @model_validator(mode="after")
    def _database_urls_must_be_postgres(self) -> BackupSettings:
        for name, value in (
            ("DATABASE_URL", self.database_url),
            ("RESTORE_CHECK_DATABASE_URL", self.restore_check_database_url),
        ):
            try:
                url = make_url(value)
            except ArgumentError:
                raise ValueError(f"{name} must be a postgresql:// URL") from None
            if url.get_backend_name() != "postgresql" or not url.database:
                raise ValueError(f"{name} must be a postgresql:// URL naming a database")
        return self

    # Checked up front: otherwise a missing directory surfaces only after the
    # dump has run for as long as a dump takes.
    @model_validator(mode="after")
    def _spool_dir_must_be_writable(self) -> BackupSettings:
        if not self.backup_spool_dir.is_dir():
            raise ValueError("BACKUP_SPOOL_DIR must be an existing directory")
        if not os.access(self.backup_spool_dir, os.W_OK | os.X_OK):
            raise ValueError("BACKUP_SPOOL_DIR must be writable by this user")
        return self

    @model_validator(mode="after")
    def _remote_must_be_name_and_path(self) -> BackupSettings:
        if not _REMOTE.fullmatch(self.backup_rclone_remote):
            raise ValueError(
                "BACKUP_RCLONE_REMOTE must have the form name:path, naming an "
                "rclone crypt remote and a non-empty path in it"
            )
        return self

    # 0 daily would prune the backup just taken. The ceilings are misconfiguration
    # guards: every kept dump costs its full size in the spool and on the remote.
    @model_validator(mode="after")
    def _retention_must_be_in_range(self) -> BackupSettings:
        if not 1 <= self.backup_retention_daily <= 90:
            raise ValueError(
                "BACKUP_RETENTION_DAILY must be in range 1..90, got "
                f"{self.backup_retention_daily}; 0 would prune the backup just taken"
            )
        if not 0 <= self.backup_retention_weekly <= 52:
            raise ValueError(
                "BACKUP_RETENTION_WEEKLY must be in range 0..52, got "
                f"{self.backup_retention_weekly}; 0 keeps no weekly backups"
            )
        return self

    @model_validator(mode="after")
    def _heartbeat_url_must_be_http(self) -> BackupSettings:
        if self.backup_heartbeat_url is None:
            return self
        parsed = urlparse(self.backup_heartbeat_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                "BACKUP_HEARTBEAT_URL must be an absolute http:// or https:// URL "
                "with a host; leave it unset rather than empty"
            )
        return self


def describe_settings_error(exc: ValidationError) -> str:
    """One line per problem, safe to print: messages and env var names only.

    A missing field's location is the field name, which is the env var in lower
    case; a validator's message already names its env var.
    """
    problems = []
    for error in exc.errors(include_input=False, include_url=False, include_context=False):
        location = ".".join(str(part).upper() for part in error["loc"])
        problems.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(problems)
