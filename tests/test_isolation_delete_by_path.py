"""The tenant boundary for DELETE /api/v1/notes?... and DELETE /api/v1/scopes.

Same two rules as the other isolation modules: a foreign caller's request must
(1) leave the owner's notes AND passages untouched -- passages are asserted
too, since an unfiltered DELETE takes them through the composite FK's cascade
-- and (2) be byte-identical to the answer for "no such note", so the response
is not an existence oracle for another tenant's scopes or paths.
"""
from sqlalchemy import select

from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key, make_note, make_passage


def _seed(
    db_session,
    client_name,
    scope="shared",
    path_or_uri="docs/contested.md",
    passages=2,
):
    note = make_note(db_session, client_name, scope=scope, path_or_uri=path_or_uri)
    ids = {
        make_passage(
            db_session, note, ordinal=o, content=f"{client_name} passage {o}"
        ).id
        for o in range(passages)
    }
    return note.id, ids


def _stored(db_session, client_name):
    db_session.expire_all()
    notes = set(
        db_session.scalars(select(Note.id).where(Note.client_name == client_name)).all()
    )
    passages = set(
        db_session.scalars(
            select(Passage.id).where(Passage.client_name == client_name)
        ).all()
    )
    return notes, passages


def _by_path(client, key, scope, path_or_uri):
    return client.delete(
        "/api/v1/notes",
        params={"scope": scope, "path_or_uri": path_or_uri},
        headers={"X-API-Key": key},
    )


def _scope(client, key, **params):
    return client.delete("/api/v1/scopes", params=params, headers={"X-API-Key": key})


def test_delete_by_path_cannot_reach_another_tenants_note(client, db_session):
    acme_note, acme_passages = _seed(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    response = _by_path(client, globex_key, "shared", "docs/contested.md")

    assert response.status_code == 404
    assert _stored(db_session, "acme") == ({acme_note}, acme_passages)


def test_foreign_delete_by_path_is_identical_to_no_such_note(client, db_session):
    _seed(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    foreign = _by_path(client, globex_key, "shared", "docs/contested.md")
    missing = _by_path(client, globex_key, "shared", "docs/never-existed.md")

    assert foreign.status_code == missing.status_code == 404
    assert foreign.text == missing.text
    assert foreign.json() == {"detail": "Note not found"}


def test_the_client_filter_is_what_prevents_the_delete_by_path(client, db_session):
    """Foreign attempt first, owner second: the owner's 200 afterwards proves
    the (scope, path) was deletable all along, so the 404 was the filter."""
    acme_note, _ = _seed(db_session, "acme")
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    assert (
        _by_path(client, globex_key, "shared", "docs/contested.md").status_code == 404
    )
    assert _stored(db_session, "acme")[0] == {acme_note}

    owner = _by_path(client, acme_key, "shared", "docs/contested.md")
    assert owner.status_code == 200
    assert owner.json()["notes"][0]["note_id"] == acme_note
    assert _stored(db_session, "acme") == (set(), set())


def test_same_scope_and_path_in_two_tenants_deletes_only_the_callers(
    client, db_session
):
    acme_note, acme_passages = _seed(db_session, "acme")
    globex_note, _ = _seed(db_session, "globex")
    globex_key = make_api_key(db_session, "globex")

    response = _by_path(client, globex_key, "shared", "docs/contested.md")

    assert response.status_code == 200
    assert response.json()["notes"][0]["note_id"] == globex_note
    assert _stored(db_session, "globex") == (set(), set())
    assert _stored(db_session, "acme") == ({acme_note}, acme_passages)


def test_delete_scope_cannot_reach_another_tenants_scope(client, db_session):
    acme_note, acme_passages = _seed(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    response = _scope(client, globex_key, scope="shared")

    assert response.status_code == 200
    assert response.json() == {"notes_deleted": 0, "passages_deleted": 0, "notes": []}
    assert _stored(db_session, "acme") == ({acme_note}, acme_passages)


def test_foreign_delete_scope_is_identical_to_no_such_scope(client, db_session):
    _seed(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    foreign = _scope(client, globex_key, scope="shared")
    missing = _scope(client, globex_key, scope="never-existed")

    assert foreign.status_code == missing.status_code == 200
    assert foreign.text == missing.text


def test_root_scope_delete_cannot_reach_another_tenants_root_scope(
    client, db_session
):
    acme_note, acme_passages = _seed(db_session, "acme", scope="")
    _, globex_passages = _seed(db_session, "globex", scope="")
    globex_key = make_api_key(db_session, "globex")

    response = _scope(client, globex_key, root_scope="true")

    assert response.status_code == 200
    assert response.json()["notes_deleted"] == 1
    assert response.json()["passages_deleted"] == len(globex_passages)
    assert _stored(db_session, "globex") == (set(), set())
    assert _stored(db_session, "acme") == ({acme_note}, acme_passages)


def test_passages_deleted_never_counts_another_tenants_passages(client, db_session):
    _seed(db_session, "acme", passages=5)
    _, globex_passages = _seed(db_session, "globex", passages=2)
    globex_key = make_api_key(db_session, "globex")

    by_path = _by_path(client, globex_key, "shared", "docs/contested.md")
    assert by_path.json()["passages_deleted"] == len(globex_passages)


def test_delete_by_id_still_cannot_cross_tenants(client, db_session):
    acme_note, acme_passages = _seed(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    response = client.delete(
        f"/api/v1/notes/{acme_note}", headers={"X-API-Key": globex_key}
    )

    assert response.status_code == 404
    assert _stored(db_session, "acme") == ({acme_note}, acme_passages)
