"""deploy/kubernetes/nescio-backup-cronjob.yaml: the properties that would
silently break recovery if they drifted, not YAML trivia.
"""
from pathlib import Path

import yaml

from scripts.backup_settings import BackupSettings

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "deploy" / "kubernetes" / "nescio-backup-cronjob.yaml"

LITERAL_ENV_FORBIDDEN = {"DATABASE_URL", "RESTORE_CHECK_DATABASE_URL", "BACKUP_HEARTBEAT_URL"}


def _load() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _job_spec(doc: dict) -> dict:
    return doc["spec"]["jobTemplate"]["spec"]


def _pod_spec(doc: dict) -> dict:
    return _job_spec(doc)["template"]["spec"]


def _container(doc: dict) -> dict:
    containers = _pod_spec(doc)["containers"]
    assert len(containers) == 1
    return containers[0]


def test_concurrency_is_forbidden():
    assert _load()["spec"]["concurrencyPolicy"] == "Forbid"


def test_a_failed_run_is_never_retried():
    # A retry would delay the missing success, which is the alert.
    assert _job_spec(_load())["backoffLimit"] == 0


def test_a_run_is_bounded_in_time():
    assert _job_spec(_load())["activeDeadlineSeconds"] > 0


def test_a_missed_schedule_has_a_deadline():
    assert _load()["spec"]["startingDeadlineSeconds"] > 0


def _default_spool() -> str:
    # .as_posix(): BackupSettings.backup_spool_dir is a Path, which on this
    # platform renders with backslashes, but the manifest's mountPath is
    # always a POSIX path inside the container.
    return BackupSettings.model_fields["backup_spool_dir"].default.as_posix()


def test_the_spool_mount_matches_the_settings_default():
    doc = _load()
    mounts = {m["name"]: m["mountPath"] for m in _container(doc)["volumeMounts"]}
    assert _default_spool() in mounts.values()


def test_the_spool_is_an_emptydir_with_a_size_limit():
    doc = _load()
    volumes = {v["name"]: v for v in _pod_spec(doc)["volumes"]}
    mounts = {m["mountPath"]: m["name"] for m in _container(doc)["volumeMounts"]}
    spool_volume_name = mounts[_default_spool()]
    spool_volume = volumes[spool_volume_name]
    assert "emptyDir" in spool_volume
    assert spool_volume["emptyDir"]["sizeLimit"]


def test_the_root_filesystem_is_read_only():
    assert _container(_load())["securityContext"]["readOnlyRootFilesystem"] is True


def test_the_container_runs_as_the_unprivileged_uid():
    doc = _load()
    pod_security = _pod_spec(doc)["securityContext"]
    assert pod_security["runAsUser"] == 10001
    assert pod_security.get("runAsNonRoot") is True


def test_no_database_or_heartbeat_url_is_a_literal_env_value():
    doc = _load()
    container = _container(doc)
    for entry in container.get("env", []):
        assert entry["name"] not in LITERAL_ENV_FORBIDDEN, (
            f"{entry['name']} must come from a Secret, never a literal value in the manifest"
        )
        if entry["name"] in LITERAL_ENV_FORBIDDEN:
            assert "value" not in entry


def test_the_backup_env_comes_from_a_secret():
    doc = _load()
    container = _container(doc)
    secret_names = {ref["secretRef"]["name"] for ref in container.get("envFrom", [])}
    assert "nescio-backup-env" in secret_names


def test_the_rclone_config_comes_from_a_secret():
    doc = _load()
    volumes = {v["name"]: v for v in _pod_spec(doc)["volumes"]}
    secret_names = {v["secret"]["secretName"] for v in volumes.values() if "secret" in v}
    assert "nescio-backup-rclone" in secret_names
