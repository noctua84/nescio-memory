"""Ingest behaviour: persistence, replacement, and input validation."""
from sqlalchemy import func, select

from app.models.learning import Learning
from tests.factories import make_api_key

LONG_ENOUGH = (
    "This content comfortably exceeds the fifty character minimum that the "
    "ingest endpoint applies to each chunk."
)


def _ingest(client, key, **overrides):
    payload = {
        "repo_name": "repo_a",
        "file_path": "docs/note.md",
        "content": LONG_ENOUGH,
    }
    payload.update(overrides)
    return client.post("/api/v1/ingest", data=payload, headers={"X-API-Key": key})


def test_ingest_persists_a_chunk(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key)

    assert response.status_code == 200
    body = response.json()
    # Aliases, not field names: IngestResponse sets serialization_alias and
    # FastAPI serialises by_alias.
    assert body["status"] == "success"
    assert body["file"] == "docs/note.md"
    assert body["ingested"] == 1

    db_session.expire_all()
    rows = db_session.scalars(select(Learning)).all()
    assert len(rows) == 1
    assert rows[0].content == LONG_ENOUGH
    assert rows[0].repo_name == "repo_a"
    assert rows[0].meta["relative_path"] == "docs/note.md"
    assert rows[0].meta["file_name"] == "note.md"
    assert rows[0].meta["chunk_index"] == 0
    assert len(rows[0].embedding) == 384


def test_reingesting_the_same_file_replaces_rather_than_duplicates(client, db_session):
    key = make_api_key(db_session, "acme")

    _ingest(client, key, content=LONG_ENOUGH)
    _ingest(client, key, content=LONG_ENOUGH + " Revised.")

    db_session.expire_all()
    rows = db_session.scalars(select(Learning)).all()
    assert len(rows) == 1
    assert rows[0].content.endswith("Revised.")


def test_reingesting_does_not_disturb_another_file_in_the_same_repo(client, db_session):
    key = make_api_key(db_session, "acme")

    _ingest(client, key, file_path="docs/one.md")
    _ingest(client, key, file_path="docs/two.md")
    _ingest(client, key, file_path="docs/one.md", content=LONG_ENOUGH + " Again.")

    db_session.expire_all()
    paths = db_session.scalars(select(Learning.file_path)).all()
    assert sorted(paths) == ["docs/one.md", "docs/two.md"]


def test_content_shorter_than_the_minimum_is_skipped(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="too short to keep")

    assert response.status_code == 200
    assert response.json()["ingested"] == 0
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_long_content_is_split_into_several_chunks(client, db_session):
    # chunk_size 1000 with overlap 200 advances 800 characters per chunk, so
    # 2000 characters yields 3 windows.
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * 2000)

    assert response.json()["ingested"] == 3
    db_session.expire_all()
    rows = db_session.scalars(select(Learning)).all()
    assert len(rows) == 3
    # chunk_index is what a consumer uses to reassemble the file in order, so
    # assert the actual values rather than just the count.
    assert sorted(row.meta["chunk_index"] for row in rows) == [0, 1, 2]


def test_an_absolute_file_path_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, file_path="/etc/passwd")

    assert response.status_code == 400
    assert "relative path" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_a_traversing_file_path_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, file_path="docs/../../secrets.md")

    assert response.status_code == 400
    assert "relative path" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_validation_runs_before_any_deletion(client, db_session):
    # _validate_file_path is called before delete_by_file. A rejected path must
    # not have destroyed an existing row on its way out.
    key = make_api_key(db_session, "acme")
    _ingest(client, key, file_path="docs/note.md")

    _ingest(client, key, file_path="/etc/passwd")

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 1


def test_content_over_the_cap_is_rejected(client, db_session):
    from app.api.v1.ingest import MAX_CONTENT_CHARS

    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * (MAX_CONTENT_CHARS + 1))

    assert response.status_code == 400
    assert "too large" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_content_exactly_at_the_cap_is_accepted(client, db_session):
    # Boundary: the cap is inclusive, so exactly MAX_CONTENT_CHARS must pass.
    from app.api.v1.ingest import MAX_CONTENT_CHARS

    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * MAX_CONTENT_CHARS)

    assert response.status_code == 200
    assert response.json()["ingested"] > 0


def test_the_cap_is_checked_before_any_deletion(client, db_session):
    # Oversized content must not destroy what is already stored.
    key = make_api_key(db_session, "acme")
    _ingest(client, key, file_path="docs/note.md")

    from app.api.v1.ingest import MAX_CONTENT_CHARS

    _ingest(
        client,
        key,
        file_path="docs/note.md",
        content="x" * (MAX_CONTENT_CHARS + 1),
    )

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 1
