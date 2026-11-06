"""Retention selection, and the pruning built on it. No database and no rclone."""
from datetime import UTC, datetime, timedelta

import pytest

from scripts._backup_common import (
    BackupError,
    RcloneRemote,
    backup_stem,
    parse_backup_name,
)
from scripts.backup import plan_prune, prune_remote, prune_spool, select_retained
from tests._backup_helpers import REMOTE, FakeRclone

# A Monday, so ISO week boundaries in these tests are easy to count from.
MONDAY = datetime(2026, 11, 2, 3, 30, tzinfo=UTC)


def _nightly(count: int, newest: datetime = MONDAY) -> list[datetime]:
    return [newest - timedelta(days=age) for age in range(count)]


def _names(*moments: datetime) -> list[str]:
    return [
        f"{backup_stem(moment)}{suffix}"
        for moment in moments
        for suffix in (".dump", ".manifest.json")
    ]


def test_no_backups_retains_nothing():
    assert select_retained([], daily=14, weekly=8) == set()


def test_daily_keeps_the_most_recent_days_and_no_more():
    nightly = _nightly(10)
    assert select_retained(nightly, daily=3, weekly=0) == set(nightly[:3])


def test_two_backups_on_one_day_keep_only_the_newer():
    morning = MONDAY.replace(hour=1)
    evening = MONDAY.replace(hour=20)
    assert select_retained([morning, evening], daily=1, weekly=0) == {evening}


def test_weekly_keeps_the_newest_of_each_iso_week_beyond_the_daily_window():
    nightly = _nightly(30)
    retained = select_retained(nightly, daily=1, weekly=3)
    # This Monday alone is the newest of its week and also the one daily; then the
    # Sundays closing the two weeks before it.
    assert retained == {MONDAY, MONDAY - timedelta(days=1), MONDAY - timedelta(days=8)}


def test_the_stated_policy_keeps_at_most_fourteen_plus_eight():
    nightly = _nightly(200)
    retained = select_retained(nightly, daily=14, weekly=8)
    assert set(nightly[:14]) <= retained
    assert 14 < len(retained) <= 22
    # Eight weeks back is kept, far older is not.
    assert min(retained) >= MONDAY - timedelta(weeks=8)


def test_weekly_zero_keeps_daily_only():
    nightly = _nightly(30)
    assert select_retained(nightly, daily=14, weekly=0) == set(nightly[:14])


def test_old_backups_do_not_age_out_when_backups_stop():
    # The selection counts backups, not the calendar: after a long outage the last
    # good backups are the most valuable ones.
    ancient = _nightly(5, newest=MONDAY - timedelta(days=4000))
    assert select_retained(ancient, daily=14, weekly=8) == set(ancient)


@pytest.mark.parametrize("daily,weekly", [(1, 0), (1, 52), (14, 8), (90, 0)])
def test_the_newest_backup_is_always_kept(daily, weekly):
    nightly = _nightly(100)
    assert max(nightly) in select_retained(nightly, daily=daily, weekly=weekly)


def test_plan_prune_deletes_a_whole_backup_never_half_of_one():
    keep, drop = MONDAY, MONDAY - timedelta(days=1)
    assert plan_prune(_names(keep, drop), daily=1, weekly=0) == sorted(_names(drop))


def test_plan_prune_never_touches_a_name_it_did_not_write():
    foreign = [
        "notes.txt",
        "nescio-latest.dump",
        "nescio-20261102T033000Z.dump.partial",
        "nescio-20261399T033000Z.dump",
    ]
    assert plan_prune([*foreign, *_names(MONDAY)], daily=1, weekly=0) == []


def test_half_a_backup_takes_no_slot_and_is_deleted():
    complete = MONDAY - timedelta(days=1)
    orphan = f"{backup_stem(MONDAY)}.dump"
    assert plan_prune([orphan, *_names(complete)], daily=1, weekly=0) == [orphan]


def test_backup_names_are_flat_short_and_round_trip():
    name = f"{backup_stem(MONDAY)}.manifest.json"
    assert "/" not in name
    assert len(name) < 143
    assert parse_backup_name(name) == (MONDAY, ".manifest.json")


def test_prune_spool_deletes_the_planned_files_only(tmp_path):
    keep, drop = MONDAY, MONDAY - timedelta(days=1)
    for name in [*_names(keep, drop), "unrelated.txt"]:
        (tmp_path / name).write_text("x")
    (tmp_path / ".restore-check").mkdir()

    assert prune_spool(tmp_path, daily=1, weekly=0) == 2

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [*_names(keep), "unrelated.txt", ".restore-check"]
    )


def test_prune_remote_deletes_one_file_at_a_time_and_never_syncs():
    keep, drop = MONDAY, MONDAY - timedelta(days=1)
    rclone = FakeRclone({name: b"x" for name in _names(keep, drop)})

    assert prune_remote(RcloneRemote(rclone, REMOTE), _names(keep), daily=1, weekly=0) == 2

    assert sorted(rclone.files) == sorted(_names(keep))
    assert set(rclone.commands()) == {"lsjson", "deletefile"}


def test_prune_remote_refuses_a_listing_without_the_new_backup():
    old = _names(MONDAY - timedelta(days=1), MONDAY - timedelta(days=2))
    rclone = FakeRclone({name: b"x" for name in old})
    remote = RcloneRemote(rclone, REMOTE)
    not_listed = _names(MONDAY)

    with pytest.raises(BackupError, match="refusing to prune"):
        prune_remote(remote, not_listed, daily=1, weekly=0)

    assert sorted(rclone.files) == sorted(old)
