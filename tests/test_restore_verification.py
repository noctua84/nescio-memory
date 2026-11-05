"""The verification functions, on a database restored with the real pg_restore.

The whole pipeline, with the heartbeat, is in tests/test_restore_check.py.
"""
import json
import re
from collections import Counter
from itertools import count
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.config import settings as app_settings
from app.repositories.passage_search import PassageSearchRepository
from scripts import restore_verification
from scripts._backup_common import Manifest, make_engine
from scripts.restore_check import (
    create_scratch_database,
    drop_scratch_database,
    restore_dump,
)
from scripts.restore_verification import (
    HNSW_INDEX,
    VerificationError,
    assert_alembic_revision,
    assert_client_counts,
    assert_hnsw_index_valid,
    assert_search_finds_a_stored_passage,
    force_hnsw_plan,
    verify_restored_database,
)
from tests._backup_helpers import (
    SEEDED_CLIENTS,
    ContainerTools,
    backed_up,  # noqa: F401  (fixture)
    source_url,  # noqa: F401  (fixture)
)
from tests.factories import make_note, make_passage
from tests.fakes import fake_embedding
from tests.test_search_index_usage import walk

SCRATCH = "restore_check_test"

# seed_source's passages less its one zero vector. Below the sample size, so a run
# on the seed alone always draws every one of them, whatever the random order.
SEEDED_ELIGIBLE = sum(client["passages"] for client in SEEDED_CLIENTS.values()) - 1
FILLER = "filler passage"
NO_SUCH_NOTE = -1
ARMS = ["search", "index"]


@pytest.fixture
def pg_tools(pg_container):
    return ContainerTools(pg_container)


@pytest.fixture
def restored(backed_up, pg_container, pg_tools, tmp_path):  # noqa: F811
    """A session on a scratch database restored from the real backup, and its manifest."""
    dump_name = next(name for name in backed_up if name.endswith(".dump"))
    manifest_name = dump_name.removesuffix(".dump") + ".manifest.json"
    dump_path = tmp_path / dump_name
    dump_path.write_bytes(backed_up[dump_name])
    url = pg_container.get_connection_url()

    admin = make_engine(url)
    create_scratch_database(admin, SCRATCH)
    restore_dump(pg_tools, dump_path, url, SCRATCH)
    scratch = make_engine(url, SCRATCH)
    db = Session(scratch)

    yield db, Manifest.from_json(backed_up[manifest_name].decode("utf-8"))

    db.close()
    scratch.dispose()
    drop_scratch_database(admin, SCRATCH)
    admin.dispose()


def test_verification_passes_on_a_good_restore(restored):
    db, manifest = restored
    verify_restored_database(db, manifest)


def test_verification_fails_on_a_count_mismatch(restored):
    db, manifest = restored
    db.execute(text("DELETE FROM notes WHERE path_or_uri = 'plain.md'"))
    db.commit()
    with pytest.raises(VerificationError, match="counts differ.*acme"):
        verify_restored_database(db, manifest)


def test_verification_fails_on_a_revision_mismatch(restored):
    db, manifest = restored
    with pytest.raises(VerificationError, match="alembic revision"):
        verify_restored_database(db, manifest.model_copy(update={"alembic_revision": "nope"}))


def test_verification_fails_when_the_hnsw_index_is_missing(restored):
    # Nothing else catches this: with no index both arms answer from an exact scan
    # and pass, so the catalog check is the only thing standing here.
    db, manifest = restored
    db.execute(text(f"DROP INDEX {HNSW_INDEX}"))
    db.commit()
    with pytest.raises(VerificationError, match="missing"):
        verify_restored_database(db, manifest)


def test_an_index_of_the_right_name_and_the_wrong_kind_does_not_count(restored):
    db, _ = restored
    db.execute(text(f"DROP INDEX {HNSW_INDEX}"))
    db.execute(text(f"CREATE INDEX {HNSW_INDEX} ON passages (ordinal)"))
    db.commit()
    with pytest.raises(VerificationError, match="missing"):
        assert_hnsw_index_valid(db)


def test_verification_fails_when_the_hnsw_index_is_invalid(restored):
    db, manifest = restored
    db.execute(
        text(f"UPDATE pg_index SET indisvalid = false WHERE indexrelid = '{HNSW_INDEX}'::regclass")
    )
    db.commit()
    with pytest.raises(VerificationError, match="not valid"):
        verify_restored_database(db, manifest)


def test_the_search_fails_when_the_top_hit_is_another_note(restored, monkeypatch):
    db, _ = restored
    real_search = PassageSearchRepository.search

    def wrong_note(self, query, query_vector, top_k, repo_filter=None):
        rows = real_search(self, query, query_vector, top_k, repo_filter)
        assert rows
        return [type("Row", (), {"note_id": rows[0].note_id + 1000})()]

    monkeypatch.setattr(PassageSearchRepository, "search", wrong_note)
    with pytest.raises(VerificationError, match="top hit"):
        assert_search_finds_a_stored_passage(db)


def test_the_search_fails_when_nothing_comes_back(restored, monkeypatch):
    db, _ = restored
    monkeypatch.setattr(PassageSearchRepository, "search", lambda self, *a, **k: [])
    with pytest.raises(VerificationError, match="top hit"):
        assert_search_finds_a_stored_passage(db)


def test_a_restore_with_no_passages_cannot_pass(restored):
    # An empty dump restores perfectly and proves nothing about embeddings.
    db, _ = restored
    db.execute(text("DELETE FROM passages"))
    db.commit()
    with pytest.raises(VerificationError, match="no passage to search for"):
        assert_search_finds_a_stored_passage(db)


def test_vector_weight_zero_is_refused_by_name(restored, monkeypatch):
    db, _ = restored
    monkeypatch.setattr(app_settings, "vector_weight", 0.0)
    with pytest.raises(VerificationError, match="VECTOR_WEIGHT"):
        assert_search_finds_a_stored_passage(db)


def test_a_wrong_embedding_dimension_is_refused_by_name(restored, monkeypatch):
    db, _ = restored
    monkeypatch.setattr(app_settings, "embedding_dimension", 384)
    with pytest.raises(VerificationError, match="EMBEDDING_DIMENSION"):
        assert_search_finds_a_stored_passage(db)


def test_the_index_probe_really_reads_the_hnsw_index(restored):
    # Unforced, this table is small enough that an exact scan would answer, and the
    # check would pass with a broken index.
    db, _ = restored
    force_hnsw_plan(db)
    plan = db.execute(
        text(f"EXPLAIN (FORMAT JSON) {restore_verification._INDEX_PROBE_SQL.text}"),
        {"client_name": "acme", "embedding": str(fake_embedding("first plain passage"))},
    ).scalar_one()
    assert HNSW_INDEX in [node.get("Index Name") for node in walk(plan[0]["Plan"])]


def test_a_stored_passage_the_index_cannot_find_fails(restored, monkeypatch):
    # The production search may answer from an exact scan, so the index gets its
    # own question. Here it answers with another note.
    db, _ = restored
    monkeypatch.setattr(
        restore_verification,
        "_INDEX_PROBE_SQL",
        text("SELECT 0 WHERE :client_name <> '' AND :embedding <> ''"),
    )
    with pytest.raises(VerificationError, match="did not return"):
        assert_search_finds_a_stored_passage(db)


def test_a_restore_gets_planner_statistics_before_the_search(restored):
    db, _ = restored
    column_statistics = text("SELECT count(*) FROM pg_stats WHERE tablename = 'passages'")
    assert db.execute(column_statistics).scalar_one() == 0

    assert_search_finds_a_stored_passage(db)

    assert db.execute(column_statistics).scalar_one() > 0


def test_generated_columns_are_recomputed_not_copied(restored, backed_up, pg_tools, tmp_path):  # noqa: F811
    # Settled here rather than assumed: the archive's COPY for passages omits the
    # STORED generated column, so the restore computes it from the restored vector.
    db, _ = restored
    dump_name = next(name for name in backed_up if name.endswith(".dump"))
    container_path = pg_tools.push(tmp_path / dump_name)
    _, listing = pg_tools.run(["pg_restore", "--data-only", "-f", "-", container_path])
    copy_line = next(
        line for line in listing.decode("utf-8").splitlines()
        if line.startswith("COPY public.passages ")
    )
    assert "embedding" in copy_line
    assert "embedding_degenerate" not in copy_line

    flags = dict(
        db.execute(text("SELECT content, embedding_degenerate FROM passages")).all()
    )
    assert flags["zero vector"] is True
    assert flags["first plain passage"] is False


def _reported(capsys) -> dict[str, int]:
    """The counts on the line a successful check prints."""
    line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("restore ")
    )
    return {name: int(value) for name, value in re.findall(r"(\w+)=(\d+)", line)}


def _note_id(db, path: str) -> int:
    return db.execute(
        text("SELECT id FROM notes WHERE path_or_uri = :path"), {"path": path}
    ).scalar_one()


def _stored_vector(db, note_id: int) -> str:
    return db.execute(
        text("SELECT CAST(embedding AS text) FROM passages WHERE note_id = :id ORDER BY ordinal"),
        {"id": note_id},
    ).scalars().first()


def test_a_twin_note_with_the_identical_vector_is_not_a_miss(restored):
    db, _ = restored
    twin_a, twin_b = _note_id(db, "twin_a.md"), _note_id(db, "twin_b.md")
    stored = _stored_vector(db, twin_a)
    assert restore_verification._is_miss(db, twin_b, twin_a, "acme", stored) is False


def test_another_note_with_a_different_vector_is_a_miss(restored):
    db, _ = restored
    twin_a, plain = _note_id(db, "twin_a.md"), _note_id(db, "plain.md")
    stored = _stored_vector(db, twin_a)
    assert restore_verification._is_miss(db, plain, twin_a, "acme", stored) is True


def test_an_identical_vector_only_counts_within_the_same_client(restored):
    db, _ = restored
    twin_a, twin_b = _note_id(db, "twin_a.md"), _note_id(db, "twin_b.md")
    stored = _stored_vector(db, twin_a)
    assert restore_verification._is_miss(db, twin_b, twin_a, "globex", stored) is True


def test_the_probes_own_note_is_a_hit_and_no_top_hit_is_a_miss(restored):
    db, _ = restored
    twin_a = _note_id(db, "twin_a.md")
    stored = _stored_vector(db, twin_a)
    assert restore_verification._is_miss(db, twin_a, twin_a, "acme", stored) is False
    assert restore_verification._is_miss(db, None, twin_a, "acme", stored) is True


def test_both_arms_count_a_real_twin_answer_as_a_hit(restored, monkeypatch, capsys):
    # Two probes ask the same question and get the same note back, so one of them
    # is answered by its twin, in each arm, through the real search and index.
    db, _ = restored
    twins = {_note_id(db, "twin_a.md"), _note_id(db, "twin_b.md")}
    real_is_miss = restore_verification._is_miss
    answers = []

    def recording(session, top_note_id, probe_note_id, client_name, stored):
        miss = real_is_miss(session, top_note_id, probe_note_id, client_name, stored)
        answers.append((top_note_id, probe_note_id, miss))
        return miss

    monkeypatch.setattr(restore_verification, "_is_miss", recording)
    assert_search_finds_a_stored_passage(db)

    # Every probe answers once per arm, the search arm first.
    assert len(answers) == 2 * SEEDED_ELIGIBLE
    for arm in (answers[:SEEDED_ELIGIBLE], answers[SEEDED_ELIGIBLE:]):
        twin_answers = [
            miss for top, probe, miss in arm
            if top != probe and top in twins and probe in twins
        ]
        assert twin_answers == [False]
    reported = _reported(capsys)
    assert (reported["search_misses"], reported["index_misses"]) == (0, 0)


@pytest.fixture
def hundred_probes(restored):
    """Exactly 100 eligible passages, so the real SEARCH_PROBE_COUNT draws all of them.

    Seeded rather than lowering the constants: the seed's handful of probes cannot
    land on 25%, and 24 and 25 of 100 pin the production constants themselves.
    Flushed, not committed: the check runs on this same session.

    All in one note, so an approximate answer among them is still a hit and only
    the misses a test arranges are counted, whatever HNSW does on this draw.
    """
    db, _ = restored
    note = make_note(db, "acme", path_or_uri="filler.md")
    for ordinal in range(100 - SEEDED_ELIGIBLE):
        make_passage(db, note, ordinal=ordinal, content=f"{FILLER} {ordinal}")
    return db


def _make_the_search_miss(monkeypatch, misses: int) -> None:
    # The first calls, not chosen probes: the draw decides nothing about the count.
    real_search = PassageSearchRepository.search
    calls = count()

    def search(self, query, query_vector, top_k, repo_filter=None):
        rows = real_search(self, query, query_vector, top_k, repo_filter)
        return [SimpleNamespace(note_id=NO_SUCH_NOTE)] if next(calls) < misses else rows

    monkeypatch.setattr(PassageSearchRepository, "search", search)


def _make_the_index_miss(monkeypatch, misses: int) -> None:
    # An exact lookup of the stored vector that misses for a fixed set of filler
    # passages, none of which shares its vector, so exactly that many probes miss.
    monkeypatch.setattr(
        restore_verification,
        "_INDEX_PROBE_SQL",
        text(
            f"""
            SELECT CASE WHEN p.id IN (
                SELECT id FROM passages WHERE content LIKE '{FILLER} %' ORDER BY id LIMIT {misses}
            ) THEN {NO_SUCH_NOTE} ELSE p.note_id END
            FROM passages p
            WHERE p.client_name = :client_name AND p.embedding = CAST(:embedding AS vector)
            ORDER BY p.id LIMIT 1
            """
        ),
    )


MAKE_ARM_MISS = {"search": _make_the_search_miss, "index": _make_the_index_miss}
ARM_FAILURE = {
    "search": r"25 of 100 .*through the search",
    "index": r"HNSW index .*25 of 100 ",
}


@pytest.mark.parametrize("arm", ARMS)
def test_one_miss_among_many_does_not_fail_the_check(hundred_probes, arm, monkeypatch, capsys):
    # What this check exists for: HNSW misses about 3.4% of stored vectors at 100k
    # passages on a healthy index, and that must not page the operator.
    MAKE_ARM_MISS[arm](monkeypatch, 1)
    assert_search_finds_a_stored_passage(hundred_probes)
    reported = _reported(capsys)
    assert reported["search_probes"] == 100
    assert reported[f"{arm}_misses"] == 1


@pytest.mark.parametrize("arm", ARMS)
def test_a_miss_share_just_below_the_limit_passes(hundred_probes, arm, monkeypatch, capsys):
    MAKE_ARM_MISS[arm](monkeypatch, 24)
    assert_search_finds_a_stored_passage(hundred_probes)
    reported = _reported(capsys)
    assert reported["search_probes"] == 100
    assert reported[f"{arm}_misses"] == 24


@pytest.mark.parametrize("arm", ARMS)
def test_a_miss_share_of_exactly_the_limit_fails(hundred_probes, arm, monkeypatch):
    MAKE_ARM_MISS[arm](monkeypatch, 25)
    with pytest.raises(VerificationError, match=ARM_FAILURE[arm]):
        assert_search_finds_a_stored_passage(hundred_probes)


@pytest.fixture
def probe_log(restored, monkeypatch):
    """Each step as it happens: ("search" or "index", client, vector), or ("force",)."""
    db, _ = restored
    log = []
    real_search = PassageSearchRepository.search
    real_force = restore_verification.force_hnsw_plan
    real_execute = db.execute

    def search(self, query, query_vector, top_k, repo_filter=None):
        log.append(("search", self.client_name, tuple(query_vector)))
        return real_search(self, query, query_vector, top_k, repo_filter)

    def force(session):
        log.append(("force",))
        real_force(session)

    def execute(statement, params=None, **kwargs):
        if statement is restore_verification._INDEX_PROBE_SQL:
            vector = tuple(json.loads(params["embedding"]))
            log.append(("index", params["client_name"], vector))
        return real_execute(statement, params, **kwargs)

    monkeypatch.setattr(PassageSearchRepository, "search", search)
    monkeypatch.setattr(restore_verification, "force_hnsw_plan", force)
    monkeypatch.setattr(db, "execute", execute)
    return log


def _probes_of(log, arm: str) -> Counter:
    return Counter(step[1:] for step in log if step[0] == arm)


def test_both_arms_probe_the_same_sample_of_the_configured_size(restored, probe_log, monkeypatch):
    # Half the eligible passages, so a run that ignored the bound would be visible
    # and the two arms could each draw a different sample.
    db, _ = restored
    monkeypatch.setattr(restore_verification, "SEARCH_PROBE_COUNT", SEEDED_ELIGIBLE // 2)
    assert_search_finds_a_stored_passage(db)
    search_probes = _probes_of(probe_log, "search")
    assert search_probes.total() == SEEDED_ELIGIBLE // 2
    assert _probes_of(probe_log, "index") == search_probes


def test_degenerate_passages_are_never_probed_in_either_arm(restored, probe_log):
    # The sample is larger than the eligible set, so every passage the query would
    # admit is drawn, and the zero vector would be among them.
    db, _ = restored
    assert_search_finds_a_stored_passage(db)
    for arm in ARMS:
        probes = _probes_of(probe_log, arm)
        assert probes.total() == SEEDED_ELIGIBLE
        assert all(any(vector) for _client, vector in probes)


def test_the_probe_sample_is_drawn_afresh(restored):
    # Fixed probes would keep testing the same corner of the graph every night.
    # 3 of 6 is 20 possible samples: 20 identical draws in a row will not happen.
    db, _ = restored
    draws = {
        frozenset(db.execute(restore_verification._PROBE_SQL, {"count": 3}).all())
        for _ in range(20)
    }
    assert len(draws) > 1


def test_every_search_runs_before_the_one_force_of_the_planner(restored, probe_log):
    # The forced settings are SET LOCAL, so they outlive force_hnsw_plan for the rest
    # of the transaction, and the hybrid search under them is a nested loop over the
    # tenant that would hit the 5 s statement timeout at scale.
    db, _ = restored
    assert_search_finds_a_stored_passage(db)
    assert [step[0] for step in probe_log] == (
        ["search"] * SEEDED_ELIGIBLE + ["force"] + ["index"] * SEEDED_ELIGIBLE
    )


def _leave_the_index_empty_but_valid(db) -> None:
    """Every row back in the table, none in the HNSW index, and the catalog none the
    wiser: indisready false is what lets the rows back in without being indexed.

    TRUNCATE rebuilds the index with nothing to read, in one transaction, where
    DELETE would need a VACUUM outside it to keep the dead rows from the build.
    """
    columns = ", ".join(
        db.execute(
            text(
                "SELECT quote_ident(attname) FROM pg_attribute "
                "WHERE attrelid = 'passages'::regclass AND attnum > 0 "
                "AND NOT attisdropped AND attgenerated = ''"
            )
        ).scalars()
    )
    set_ready = f"UPDATE pg_index SET indisready = {{}} WHERE indexrelid = '{HNSW_INDEX}'::regclass"
    db.execute(text("CREATE TEMP TABLE passages_aside ON COMMIT DROP AS SELECT * FROM passages"))
    db.execute(text("TRUNCATE passages"))
    db.execute(text(set_ready.format("false")))
    db.execute(text(f"INSERT INTO passages ({columns}) SELECT {columns} FROM passages_aside"))
    db.execute(text(set_ready.format("true")))
    db.commit()


def test_a_broken_index_the_catalog_calls_valid_is_caught_by_the_index_arm(restored):
    # The acceptance case. The table is small enough for the production search to
    # answer from an exact scan and pass, the counts match, and indisvalid is true:
    # only the question put straight to the index can tell.
    db, manifest = restored
    _leave_the_index_empty_but_valid(db)

    # Proves the setup, not the code: the rows are there and the index cannot see them.
    force_hnsw_plan(db)
    assert db.execute(
        restore_verification._INDEX_PROBE_SQL,
        {"client_name": "acme", "embedding": str(fake_embedding("first plain passage"))},
    ).first() is None
    db.rollback()

    assert_alembic_revision(db, manifest)
    assert_client_counts(db, manifest)
    assert_hnsw_index_valid(db)
    with pytest.raises(VerificationError, match=f"HNSW index {HNSW_INDEX} did not return"):
        assert_search_finds_a_stored_passage(db)


def test_success_prints_the_probe_counts_on_stdout(restored, capsys):
    db, _ = restored
    capsys.readouterr()
    assert_search_finds_a_stored_passage(db)
    assert (
        f"restore search_probes={SEEDED_ELIGIBLE} search_misses=0 index_misses=0"
        in capsys.readouterr().out.splitlines()
    )


def test_a_database_error_in_the_search_arm_is_not_counted_as_a_miss(restored, monkeypatch):
    # A statement timeout or an unreadable index must fail loudly as what it is.
    db, _ = restored
    monkeypatch.setattr(
        PassageSearchRepository,
        "search",
        lambda self, *a, **k: self.db.execute(text("SELECT 1/0")).all(),
    )
    with pytest.raises(DBAPIError, match="division by zero"):
        assert_search_finds_a_stored_passage(db)


def test_a_database_error_in_the_index_arm_is_not_counted_as_a_miss(restored, monkeypatch):
    db, _ = restored
    monkeypatch.setattr(
        restore_verification,
        "_INDEX_PROBE_SQL",
        text("SELECT 1/0 WHERE :client_name <> '' AND :embedding <> ''"),
    )
    with pytest.raises(DBAPIError, match="division by zero"):
        assert_search_finds_a_stored_passage(db)
