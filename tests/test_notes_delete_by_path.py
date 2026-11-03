"""DELETE /api/v1/notes?scope=&path_or_uri= and DELETE /api/v1/scopes.

Single-client behaviour only. The tenant boundary for both routes lives in
tests/test_isolation_delete_by_path.py, and delete by id in tests/test_notes.py.

Every test that expects a 400 or a no-op asserts the stored row counts
afterwards, not just the status: a guard that answered 400 but had already run
the DELETE would pass a status-only assertion and still have destroyed data.
"""
import pytest
from sqlalchemy import func, select

from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key, make_note, make_passage

NOTES_URL = "/api/v1/notes"
SCOPES_URL = "/api/v1/scopes"


def _auth(key):
    return {"X-API-Key": key}


def _delete_by_path(client, key, scope, path_or_uri):
    return client.delete(
        NOTES_URL,
        params={"scope": scope, "path_or_uri": path_or_uri},
        headers=_auth(key),
    )


def _delete_scope(client, key, **params):
    return client.delete(SCOPES_URL, params=params, headers=_auth(key))


def _search(client, key, q):
    return client.post(
        "/api/v1/search", json={"q": q, "top_k": 10}, headers=_auth(key)
    )


def _counts(db_session, client_name="acme"):
    """(notes, passages) stored for a client, read fresh from the database."""
    db_session.expire_all()
    notes = db_session.scalar(
        select(func.count()).select_from(Note).where(Note.client_name == client_name)
    )
    passages = db_session.scalar(
        select(func.count())
        .select_from(Passage)
        .where(Passage.client_name == client_name)
    )
    return notes, passages


def _seed(db_session, scope, path_or_uri, passages=2, client_name="acme"):
    note = make_note(
        db_session, client_name, scope=scope, path_or_uri=path_or_uri
    )
    for ordinal in range(passages):
        make_passage(
            db_session, note, ordinal=ordinal, content=f"{path_or_uri} {ordinal}"
        )
    return note


def _ingest_pelican(client, key):
    return client.post(
        "/api/v1/ingest",
        json={
            "scope": "repo_a",
            "path_or_uri": "docs/pelican.md",
            "content": "Body text about the pelicanium isotope and nothing else.\n",
        },
        headers=_auth(key),
    )


# ---------------------------------------------------------------------------
# DELETE /api/v1/notes?scope=&path_or_uri=
# ---------------------------------------------------------------------------


def test_delete_by_path_removes_the_note_and_its_passages(client, db_session):
    key = make_api_key(db_session, "acme")
    target = _seed(db_session, "repo_a", "docs/target.md", passages=3)
    _seed(db_session, "repo_a", "docs/other.md", passages=2)
    target_id, target_hash = target.id, target.content_hash

    response = _delete_by_path(client, key, "repo_a", "docs/target.md")

    assert response.status_code == 200
    assert response.json() == {
        "notes_deleted": 1,
        "passages_deleted": 3,
        "notes": [
            {
                "note_id": target_id,
                "path_or_uri": "docs/target.md",
                "content_hash": target_hash,
            }
        ],
    }
    assert _counts(db_session) == (1, 2)
    assert (
        db_session.scalars(select(Passage.id).where(Passage.note_id == target_id)).all()
        == []
    )


def test_delete_by_path_only_touches_that_scope_and_path(client, db_session):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "repo_a", "docs/same.md")
    kept = _seed(db_session, "repo_b", "docs/same.md")
    kept_id = kept.id

    assert _delete_by_path(client, key, "repo_a", "docs/same.md").status_code == 200

    db_session.expire_all()
    assert db_session.scalars(select(Note.id)).all() == [kept_id]


def test_delete_by_path_works_for_the_root_scope(client, db_session):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "", "docs/root.md")

    response = _delete_by_path(client, key, "", "docs/root.md")

    assert response.status_code == 200
    assert response.json()["notes_deleted"] == 1
    assert _counts(db_session) == (0, 0)


def test_delete_by_path_accepts_a_uri(client, db_session):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "repo_a", "https://example.com/a/b?c=d")

    response = _delete_by_path(client, key, "repo_a", "https://example.com/a/b?c=d")

    assert response.status_code == 200
    assert response.json()["notes"][0]["path_or_uri"] == "https://example.com/a/b?c=d"


def test_a_note_deleted_by_path_no_longer_appears_in_search(client, db_session):
    key = make_api_key(db_session, "acme")
    note_id = _ingest_pelican(client, key).json()["note_id"]
    before = _search(client, key, "pelicanium")
    assert [r["note_id"] for r in before.json()["results"]] == [note_id]

    response = _delete_by_path(client, key, "repo_a", "docs/pelican.md")

    assert response.status_code == 200
    assert response.json()["notes"][0]["note_id"] == note_id
    assert _search(client, key, "pelicanium").json()["results"] == []


def test_delete_by_path_with_no_match_is_a_404(client, db_session):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "repo_a", "docs/present.md")

    response = _delete_by_path(client, key, "repo_a", "docs/absent.md")

    assert response.status_code == 404
    assert response.json() == {"detail": "Note not found"}
    assert _counts(db_session) == (1, 2)


def test_deleting_by_path_twice_is_a_404_the_second_time(client, db_session):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "repo_a", "docs/once.md")

    assert _delete_by_path(client, key, "repo_a", "docs/once.md").status_code == 200
    assert _delete_by_path(client, key, "repo_a", "docs/once.md").status_code == 404


@pytest.mark.parametrize("bad_path", ["/etc/passwd", "../escape.md", "a/../b.md"])
def test_delete_by_path_rejects_an_unsafe_path_like_ingest_does(
    client, db_session, bad_path
):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "repo_a", "docs/present.md")

    response = _delete_by_path(client, key, "repo_a", bad_path)

    assert response.status_code == 400
    assert _counts(db_session) == (1, 2)


def test_delete_by_path_requires_both_parameters(client, db_session):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "repo_a", "docs/present.md")

    no_scope = client.delete(
        NOTES_URL, params={"path_or_uri": "docs/present.md"}, headers=_auth(key)
    )
    no_path = client.delete(NOTES_URL, params={"scope": "repo_a"}, headers=_auth(key))
    empty_path = _delete_by_path(client, key, "repo_a", "")

    assert no_scope.status_code == no_path.status_code == empty_path.status_code == 422
    assert _counts(db_session) == (1, 2)


def test_delete_by_path_requires_authentication(client, db_session):
    _seed(db_session, "repo_a", "docs/present.md")

    response = client.delete(
        NOTES_URL, params={"scope": "repo_a", "path_or_uri": "docs/present.md"}
    )

    assert response.status_code == 401
    assert _counts(db_session) == (1, 2)


def test_delete_by_id_is_unaffected(client, db_session):
    key = make_api_key(db_session, "acme")
    note = _seed(db_session, "repo_a", "docs/by-id.md")
    _seed(db_session, "repo_a", "docs/kept.md")

    response = client.delete(f"{NOTES_URL}/{note.id}", headers=_auth(key))

    assert response.status_code == 204
    assert _counts(db_session) == (1, 2)


# ---------------------------------------------------------------------------
# DELETE /api/v1/scopes
# ---------------------------------------------------------------------------


def test_delete_scope_removes_only_that_scope(client, db_session):
    key = make_api_key(db_session, "acme")
    first = _seed(db_session, "repo_a", "docs/one.md", passages=2)
    second = _seed(db_session, "repo_a", "docs/two.md", passages=3)
    _seed(db_session, "repo_b", "docs/one.md", passages=1)
    _seed(db_session, "", "docs/root.md", passages=1)
    expected = {
        (first.id, "docs/one.md", first.content_hash),
        (second.id, "docs/two.md", second.content_hash),
    }

    response = _delete_scope(client, key, scope="repo_a")

    assert response.status_code == 200
    body = response.json()
    assert body["notes_deleted"] == 2
    assert body["passages_deleted"] == 5
    assert {
        (n["note_id"], n["path_or_uri"], n["content_hash"]) for n in body["notes"]
    } == expected
    assert _counts(db_session) == (2, 2)
    assert set(db_session.scalars(select(Note.scope)).all()) == {"repo_b", ""}


def test_delete_scope_with_no_match_is_200_with_zeros(client, db_session):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "repo_a", "docs/one.md")

    response = _delete_scope(client, key, scope="no-such-scope")

    assert response.status_code == 200
    assert response.json() == {"notes_deleted": 0, "passages_deleted": 0, "notes": []}
    assert _counts(db_session) == (1, 2)


@pytest.mark.parametrize(
    "params",
    [{}, {"scope": ""}, {"root_scope": "false"}, {"scope": "", "root_scope": "false"}],
    ids=["omitted", "empty", "omitted-explicit-false", "empty-explicit-false"],
)
def test_an_empty_or_omitted_scope_without_root_scope_is_a_400_that_deletes_nothing(
    client, db_session, params
):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "", "docs/root.md", passages=2)
    _seed(db_session, "repo_a", "docs/one.md", passages=3)
    assert _counts(db_session) == (2, 5)

    response = _delete_scope(client, key, **params)

    assert response.status_code == 400
    assert "root_scope=true" in response.json()["detail"]
    assert _counts(db_session) == (2, 5)


@pytest.mark.parametrize(
    "params", [{"root_scope": "true"}, {"scope": "", "root_scope": "true"}]
)
def test_root_scope_true_deletes_only_the_root_scope(client, db_session, params):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "", "docs/root-a.md", passages=2)
    _seed(db_session, "", "docs/root-b.md", passages=1)
    _seed(db_session, "repo_a", "docs/one.md", passages=3)

    response = _delete_scope(client, key, **params)

    assert response.status_code == 200
    body = response.json()
    assert body["notes_deleted"] == 2
    assert body["passages_deleted"] == 3
    assert {n["path_or_uri"] for n in body["notes"]} == {
        "docs/root-a.md",
        "docs/root-b.md",
    }
    assert _counts(db_session) == (1, 3)
    assert db_session.scalars(select(Note.scope)).all() == ["repo_a"]


def test_root_scope_true_with_a_non_empty_scope_is_a_400_that_deletes_nothing(
    client, db_session
):
    key = make_api_key(db_session, "acme")
    _seed(db_session, "", "docs/root.md", passages=2)
    _seed(db_session, "repo_a", "docs/one.md", passages=3)

    response = _delete_scope(client, key, scope="repo_a", root_scope="true")

    assert response.status_code == 400
    assert "cannot be combined" in response.json()["detail"]
    assert _counts(db_session) == (2, 5)


def test_delete_scope_requires_authentication(client, db_session):
    _seed(db_session, "repo_a", "docs/one.md")

    response = client.delete(SCOPES_URL, params={"scope": "repo_a"})

    assert response.status_code == 401
    assert _counts(db_session) == (1, 2)


def test_a_deleted_scope_no_longer_appears_in_search(client, db_session):
    key = make_api_key(db_session, "acme")
    _ingest_pelican(client, key)
    assert len(_search(client, key, "pelicanium").json()["results"]) == 1

    assert _delete_scope(client, key, scope="repo_a").status_code == 200

    assert _search(client, key, "pelicanium").json()["results"] == []
