#!/bin/sh
# The CronJob's two steps, in order: a script rather than an inline `sh -c` in
# the Dockerfile or the manifest, so the "run the restore check only if the
# backup succeeded, preserve the exit code, always clean the spool" logic is
# in one place and can be read and tested on its own.
#
# set -u (not -e): `set -e` would skip the cleanup trap on a failing step,
# which is exactly the case that must not leave dumps behind.
set -u

# An emptyDir arrives group-writable, so the dumps need this to be private.
umask 077

SPOOL_DIR="${BACKUP_SPOOL_DIR:-/var/lib/nescio-backup}"

# The dumps are unencrypted personal notes. A finished pod's emptyDir can
# outlive the run until the pod is garbage-collected, so nothing that held a
# dump is left for that window regardless of how this job ends.
cleanup() {
    find "$SPOOL_DIR" -mindepth 1 -delete 2>/dev/null || true
}
trap cleanup EXIT

python -m scripts.backup
backup_status=$?

if [ "$backup_status" -ne 0 ]; then
    exit "$backup_status"
fi

# Not --scheduled: the heartbeat is optional here, because a Prometheus alert on
# this job's last success does the same job. It is still sent if configured.
python -m scripts.restore_check
exit $?
