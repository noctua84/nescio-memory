"""BackupSettings: its bounds, its secrecy, and its own README table.

tests/test_readme_config.py enforces the same rows for app.config's Settings; this
holds BackupSettings to the same rule, against the table in "Backup and recovery".
"""
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import backup_settings
from scripts.backup_settings import BackupSettings, describe_settings_error

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ENV_EXAMPLE = ROOT / ".env.example"

PASSWORD = "s3cret-pw"
HEARTBEAT = "https://hc-ping.example/uuid-is-the-credential"


def _settings(spool: Path, **overrides) -> BackupSettings:
    values = {
        "database_url": f"postgresql://app:{PASSWORD}@db:5432/nescio_memory",
        "restore_check_database_url": f"postgresql://postgres:{PASSWORD}@db:5432/postgres",
        "backup_spool_dir": spool,
        "backup_rclone_remote": "secret:nescio",
        **overrides,
    }
    return BackupSettings(_env_file=None, **values)


def test_defaults(tmp_path):
    settings = _settings(tmp_path)
    assert settings.backup_retention_daily == 14
    assert settings.backup_retention_weekly == 8
    assert settings.backup_heartbeat_url is None
    assert BackupSettings.model_fields["backup_spool_dir"].default == Path("/var/lib/nescio-backup")


@pytest.mark.parametrize(
    "field, value, env_var",
    [
        ("backup_retention_daily", 0, "BACKUP_RETENTION_DAILY"),
        ("backup_retention_daily", 91, "BACKUP_RETENTION_DAILY"),
        ("backup_retention_weekly", -1, "BACKUP_RETENTION_WEEKLY"),
        ("backup_retention_weekly", 53, "BACKUP_RETENTION_WEEKLY"),
        ("backup_rclone_remote", "", "BACKUP_RCLONE_REMOTE"),
        ("backup_rclone_remote", "secret", "BACKUP_RCLONE_REMOTE"),
        ("backup_rclone_remote", "secret:", "BACKUP_RCLONE_REMOTE"),
        ("backup_rclone_remote", "/var/backups", "BACKUP_RCLONE_REMOTE"),
        ("backup_rclone_remote", r"C:\backups", "BACKUP_RCLONE_REMOTE"),
        ("backup_rclone_remote", ":s3,access_key_id=AKIA:bucket", "BACKUP_RCLONE_REMOTE"),
        ("backup_rclone_remote", "s3,access_key_id=AKIA:bucket", "BACKUP_RCLONE_REMOTE"),
        ("backup_heartbeat_url", "", "BACKUP_HEARTBEAT_URL"),
        ("backup_heartbeat_url", "ftp://hc-ping.example/x", "BACKUP_HEARTBEAT_URL"),
        ("backup_heartbeat_url", "hc-ping.example/x", "BACKUP_HEARTBEAT_URL"),
        ("backup_heartbeat_url", "https://", "BACKUP_HEARTBEAT_URL"),
        ("database_url", "mysql://u:p@db/x", "DATABASE_URL"),
        ("database_url", "not a url", "DATABASE_URL"),
        ("database_url", "postgresql://u:p@db", "DATABASE_URL"),
        ("restore_check_database_url", "mysql://u:p@db/x", "RESTORE_CHECK_DATABASE_URL"),
        ("restore_check_database_url", "postgresql://u:p@db", "RESTORE_CHECK_DATABASE_URL"),
    ],
)
def test_a_bad_value_fails_naming_its_env_var(tmp_path, field, value, env_var):
    with pytest.raises(ValidationError, match=env_var):
        _settings(tmp_path, **{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("backup_retention_daily", 1),
        ("backup_retention_daily", 90),
        ("backup_retention_weekly", 0),
        ("backup_retention_weekly", 52),
        ("backup_rclone_remote", "secret:path/to/backups"),
        ("backup_rclone_remote", "my-crypt.remote:nescio"),
        ("backup_heartbeat_url", "http://localhost:8000/ping"),
        ("backup_heartbeat_url", HEARTBEAT),
        ("database_url", "postgresql+psycopg://u:p@db:5432/nescio_memory"),
    ],
)
def test_the_bounds_are_inclusive(tmp_path, field, value):
    assert getattr(_settings(tmp_path, **{field: value}), field) == value


def test_a_missing_spool_dir_fails_naming_its_env_var(tmp_path):
    with pytest.raises(ValidationError, match="BACKUP_SPOOL_DIR must be an existing directory"):
        _settings(tmp_path / "absent")


def test_a_file_as_spool_dir_fails(tmp_path):
    a_file = tmp_path / "file"
    a_file.write_text("x")
    with pytest.raises(ValidationError, match="BACKUP_SPOOL_DIR must be an existing directory"):
        _settings(a_file)


def test_an_unwritable_spool_dir_fails(tmp_path, monkeypatch):
    # chmod does not bind on every platform the suite runs on, or for root.
    monkeypatch.setattr(backup_settings.os, "access", lambda path, mode: False)
    with pytest.raises(ValidationError, match="BACKUP_SPOOL_DIR must be writable"):
        _settings(tmp_path)


def test_required_settings_are_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("DATABASE_URL", "BACKUP_RCLONE_REMOTE", "RESTORE_CHECK_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValidationError) as caught:
        BackupSettings(backup_spool_dir=tmp_path)
    described = describe_settings_error(caught.value)
    for name in ("DATABASE_URL", "BACKUP_RCLONE_REMOTE", "RESTORE_CHECK_DATABASE_URL"):
        assert f"{name}: Field required" in described


def test_settings_are_read_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db/source")
    monkeypatch.setenv("RESTORE_CHECK_DATABASE_URL", "postgresql://u:p@db/postgres")
    monkeypatch.setenv("BACKUP_SPOOL_DIR", str(tmp_path))
    monkeypatch.setenv("BACKUP_RCLONE_REMOTE", "secret:nescio")
    monkeypatch.setenv("BACKUP_RETENTION_DAILY", "7")
    monkeypatch.setenv("BACKUP_HEARTBEAT_URL", HEARTBEAT)
    settings = BackupSettings()
    assert settings.backup_retention_daily == 7
    assert settings.backup_heartbeat_url == HEARTBEAT


@pytest.mark.parametrize(
    "overrides",
    [
        {"backup_retention_daily": 0},
        {"backup_rclone_remote": "bad"},
        {"backup_heartbeat_url": f"ftp://hc-ping.example/{PASSWORD}"},
    ],
)
def test_no_validation_error_prints_a_secret(tmp_path, overrides):
    # pydantic prints an after-validator's whole input in str(exc) by default,
    # which here is every URL and its password.
    with pytest.raises(ValidationError) as caught:
        _settings(tmp_path, **overrides)
    assert PASSWORD not in str(caught.value)
    assert PASSWORD not in describe_settings_error(caught.value)


def test_repr_hides_the_urls(tmp_path):
    rendered = repr(_settings(tmp_path, backup_heartbeat_url=HEARTBEAT))
    assert PASSWORD not in rendered
    assert HEARTBEAT not in rendered


def _backup_section() -> str:
    text = README.read_text(encoding="utf-8")
    return text.split("\n## Backup and recovery", 1)[1].split("\n## ", 1)[0]


def _table_rows() -> dict[str, str]:
    """{env var: default cell} for the section's configuration table."""
    rows = {}
    for line in _backup_section().splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 3 and (name := re.fullmatch(r"`([A-Z][A-Z0-9_]*)`", cells[1])):
            rows[name.group(1)] = cells[2]
    return rows


def test_every_setting_has_a_row_in_the_backup_table():
    missing = [name.upper() for name in BackupSettings.model_fields if name.upper() not in _table_rows()]
    assert not missing, f"BackupSettings fields without a README row: {missing}"


def test_every_setting_has_an_env_example_line():
    present = set(
        re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    )
    missing = [name.upper() for name in BackupSettings.model_fields if name.upper() not in present]
    assert not missing, f"BackupSettings fields without a NAME= line in .env.example: {missing}"


@pytest.mark.parametrize("field", ["backup_spool_dir", "backup_retention_daily", "backup_retention_weekly"])
def test_the_readme_default_matches_the_setting(field):
    documented = _table_rows()[field.upper()]
    assert documented == f"`{BackupSettings.model_fields[field].default}`".replace("\\", "/")


def test_the_units_use_lf_line_endings():
    # A carriage return in a unit file is a parse error or a corrupted value.
    for unit in ("nescio-backup.service", "nescio-backup.timer"):
        assert b"\r" not in (ROOT / "deploy" / "systemd" / unit).read_bytes()
