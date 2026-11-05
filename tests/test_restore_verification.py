"""The verification functions, on a database restored with the real pg_restore.

The whole pipeline, with the heartbeat, is in tests/test_restore_check.py.
"""
import pytest
from sqlalchemy import text
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
    assert_hnsw_index_valid,
    assert_search_finds_a_stored_passage,
    force_hnsw_plan,
    verify_restored_database,
)
from tests._backup_helpers import (
    ContainerTools,
    backed_up,  # noqa: F401  (fixture)
    source_url,  # noqa: F401  (fixture)
)
from tests.fakes import fake_embedding
from tests.test_search_index_usage import walk

SCRATCH = "restore_check_test"


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


def test_the_probe_looks_past_an_embedding_two_notes_share(restored):
    # Their shared vector could rank either note first on a healthy restore.
    db, _ = restored
    note_id = db.execute(restore_verification._PROBE_SQL).one()[0]
    path = db.execute(
        text("SELECT path_or_uri FROM notes WHERE id = :id"), {"id": note_id}
    ).scalar_one()
    assert path == "plain.md"


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
