"""The tenant boundary for /api/v2 -- ingest, get, delete and search.

The v1 counterpart (tests/test_isolation.py) states the stakes and they are
unchanged here: every failure mode in this module is silent in production. No
exception, no error response, just one client reading or destroying another
client's notes.

Two things give an isolation test teeth, both borrowed from that module:

1. The non-owner must itself hold data at the contested identity. A tenant
   with no rows at all cannot detect a missing filter -- its query matches
   nothing either way, so the request passes against a completely unscoped
   repository. Where it is cheap, globex is given its own row at the same
   (scope, path_or_uri) or its own marker-bearing passage, so the statement
   under test really runs.
2. The pair of requests must differ ONLY in the key presented. When the URL,
   the body and the seeded rows are identical, a passing pair cannot be
   explained by anything except the `client_name` filter -- no mutation of
   production code is needed to show the filter is load-bearing.

Markers, not field names, are what the leak assertions use: each tenant's text
carries a sentinel that cannot appear in the other's, and the assertion is
against `response.text` -- the WHOLE body -- so a leak through any field,
including one added later, fails the test rather than slipping past a
field-by-field check.

v1 is untouched; nothing here modifies tests/test_isolation.py.
"""
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key, make_note, make_passage

# Single lowercase tokens, deliberately: v2 search is full text, and
# to_tsvector/websearch_to_tsquery must agree on the token for a marker to be
# both findable by the owner and assertable-absent for everyone else.
ACME_MARKER = "xenofluxacme"
GLOBEX_MARKER = "quorvaneglobex"

# The word both tenants' passages share, so one query body reaches both
# tenants' data and the filter is the only thing separating the answers.
SHARED_WORD = "contested"


def _body(marker: str) -> str:
    """A short headingless note stamped with `marker`; one passage.

    Headingless and under the splitter's TARGET_CHARS, so the body stays whole
    and the marker is guaranteed to be in the single passage that results --
    no test here depends on how the splitter would cut a larger document.
    """
    return (
        f"{marker} this {SHARED_WORD} note belongs to exactly one tenant and "
        f"its contents must never surface for another. {marker}"
    )


def _ingest(client, key, **overrides):
    payload = {
        "scope": "shared",
        "path_or_uri": "docs/note.md",
        "content": _body(ACME_MARKER),
    }
    payload.update(overrides)
    return client.post("/api/v2/ingest", json=payload, headers={"X-API-Key": key})


def _get_note(client, key, note_id):
    return client.get(f"/api/v2/notes/{note_id}", headers={"X-API-Key": key})


def _delete_note(client, key, note_id):
    return client.delete(f"/api/v2/notes/{note_id}", headers={"X-API-Key": key})


def _search(client, key, **overrides):
    payload = {"q": SHARED_WORD, "top_k": 50}
    payload.update(overrides)
    return client.post("/api/v2/search", json=payload, headers={"X-API-Key": key})


# ---------------------------------------------------------------------------
# POST /api/v2/ingest
# ---------------------------------------------------------------------------


def test_ingest_v2_stamps_the_authenticated_clients_name(client, db_session):
    # upsert_note and replace_passages stamp client_name from the repository's
    # own field; the route never passes one. Both tables are checked because
    # they are stamped by two different statements.
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key)
    assert response.status_code == 200

    db_session.expire_all()
    note_owners = db_session.scalars(select(Note.client_name)).all()
    passage_owners = db_session.scalars(select(Passage.client_name)).all()
    assert note_owners and passage_owners
    assert set(note_owners) == {"acme"}
    assert set(passage_owners) == {"acme"}


def test_ingest_v2_of_a_shared_path_does_not_touch_another_tenants_note(
    client, db_session
):
    # The upsert's ON CONFLICT target is (client_name, scope, path_or_uri).
    # Were client_name not stamped into the conflicting identity, globex's
    # ingest would overwrite acme's body in place.
    acme_note = make_note(
        db_session,
        "acme",
        scope="shared",
        path_or_uri="docs/note.md",
        body=_body(ACME_MARKER),
    )
    acme_id, acme_body, acme_hash = (
        acme_note.id,
        acme_note.body,
        acme_note.content_hash,
    )
    globex_key = make_api_key(db_session, "globex")

    response = _ingest(client, globex_key, content=_body(GLOBEX_MARKER))
    assert response.status_code == 200

    db_session.expire_all()
    survivor = db_session.scalars(
        select(Note).where(Note.client_name == "acme")
    ).one()
    assert (survivor.id, survivor.body, survivor.content_hash) == (
        acme_id,
        acme_body,
        acme_hash,
    )
    assert GLOBEX_MARKER not in survivor.body


def test_ingest_v2_of_a_shared_path_does_not_delete_another_tenants_passages(
    client, db_session
):
    """The destructive failure mode: replace_passages' DELETE.

    That delete carries its own tenant filter rather than relying on the
    composite FK. Without it, globex's ingest of the contested identity would
    take acme's passages by note_id alone -- a silent erasure inside a 200.
    """
    acme_note = make_note(
        db_session,
        "acme",
        scope="shared",
        path_or_uri="docs/note.md",
        body=_body(ACME_MARKER),
    )
    acme_passage_ids = {
        make_passage(
            db_session,
            acme_note,
            ordinal=ordinal,
            content=f"{ACME_MARKER} passage {ordinal} of the {SHARED_WORD} note.",
        ).id
        for ordinal in (0, 1, 2)
    }
    assert len(acme_passage_ids) == 3
    globex_key = make_api_key(db_session, "globex")

    response = _ingest(client, globex_key, content=_body(GLOBEX_MARKER))
    assert response.status_code == 200

    db_session.expire_all()
    surviving = set(
        db_session.scalars(
            select(Passage.id).where(Passage.client_name == "acme")
        ).all()
    )
    assert surviving == acme_passage_ids


def test_the_same_path_or_uri_coexists_under_two_clients_in_v2(client, db_session):
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    acme_response = _ingest(client, acme_key, content=_body(ACME_MARKER))
    globex_response = _ingest(client, globex_key, content=_body(GLOBEX_MARKER))

    assert acme_response.status_code == 200
    assert globex_response.status_code == 200
    assert acme_response.json()["note_id"] != globex_response.json()["note_id"]

    db_session.expire_all()
    rows = db_session.execute(
        select(Note.client_name, Note.id).where(Note.path_or_uri == "docs/note.md")
    ).all()
    assert sorted(row.client_name for row in rows) == ["acme", "globex"]
    assert len({row.id for row in rows}) == 2


# ---------------------------------------------------------------------------
# GET /api/v2/notes/{note_id}
# ---------------------------------------------------------------------------


def test_get_note_v2_returns_404_for_another_tenants_note_id(client, db_session):
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    globex_key = make_api_key(db_session, "globex")

    response = _get_note(client, globex_key, acme_note.id)

    assert response.status_code == 404
    assert ACME_MARKER not in response.text


def test_the_client_filter_is_what_withholds_the_note(client, db_session):
    """Two requests differing only in the key presented; one gets the note.

    One seeded note, one URL, requested twice. Because the requests are
    otherwise identical, the 200/404 split cannot be explained by anything
    except the `client_name` filter in get_note.
    """
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    owner_view = _get_note(client, acme_key, acme_note.id)
    other_view = _get_note(client, globex_key, acme_note.id)

    assert owner_view.status_code == 200
    assert owner_view.json()["id"] == acme_note.id
    assert ACME_MARKER in owner_view.json()["body"]
    assert other_view.status_code == 404
    assert ACME_MARKER not in other_view.text


def test_get_note_v2_404_is_identical_for_a_missing_and_a_foreign_note(
    client, db_session
):
    """No existence oracle: the two 404s must be indistinguishable.

    If a foreign id answered differently from an id that exists nowhere, the
    difference alone would let a caller enumerate another tenant's id space.
    """
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    globex_key = make_api_key(db_session, "globex")
    missing_id = acme_note.id + 1_000_000

    foreign = _get_note(client, globex_key, acme_note.id)
    missing = _get_note(client, globex_key, missing_id)

    assert foreign.status_code == missing.status_code == 404
    assert foreign.text == missing.text
    assert foreign.json() == {"detail": "Note not found"}


def test_get_note_v2_passage_count_counts_only_the_owners_passages(
    client, db_session
):
    """`passage_count` is the owner's own volume, not a global one.

    count_passages carries its own tenant filter: a cross-tenant count leaks
    the existence and size of another tenant's note through a number, which no
    marker assertion on the body would catch.
    """
    acme_note = make_note(
        db_session, "acme", scope="shared", path_or_uri="docs/note.md"
    )
    globex_note = make_note(
        db_session, "globex", scope="shared", path_or_uri="docs/note.md"
    )
    make_passage(
        db_session, acme_note, ordinal=0, content=f"{ACME_MARKER} the only one."
    )
    for ordinal in (0, 1, 2, 3):
        make_passage(
            db_session,
            globex_note,
            ordinal=ordinal,
            content=f"{GLOBEX_MARKER} passage {ordinal}.",
        )
    acme_key = make_api_key(db_session, "acme")

    response = _get_note(client, acme_key, acme_note.id)

    assert response.status_code == 200
    assert response.json()["passage_count"] == 1
    assert GLOBEX_MARKER not in response.text


# ---------------------------------------------------------------------------
# DELETE /api/v2/notes/{note_id}
# ---------------------------------------------------------------------------


def test_delete_note_v2_returns_404_for_another_tenants_note_id(client, db_session):
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    globex_key = make_api_key(db_session, "globex")

    response = _delete_note(client, globex_key, acme_note.id)

    assert response.status_code == 404
    assert ACME_MARKER not in response.text


def test_a_foreign_delete_leaves_the_note_and_its_passages_intact(client, db_session):
    """The destructive failure mode on the erasure path.

    Passages are seeded, not just the note: an unfiltered DELETE would take
    them with it through the composite FK's ON DELETE CASCADE, so a note-only
    assertion could miss the loss entirely.
    """
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    acme_passage_ids = {
        make_passage(
            db_session,
            acme_note,
            ordinal=ordinal,
            content=f"{ACME_MARKER} passage {ordinal}.",
        ).id
        for ordinal in (0, 1, 2)
    }
    globex_key = make_api_key(db_session, "globex")

    assert _delete_note(client, globex_key, acme_note.id).status_code == 404

    db_session.expire_all()
    assert db_session.scalars(
        select(Note.id).where(Note.client_name == "acme")
    ).all() == [acme_note.id]
    surviving = set(
        db_session.scalars(
            select(Passage.id).where(Passage.client_name == "acme")
        ).all()
    )
    assert surviving == acme_passage_ids


def test_the_client_filter_is_what_prevents_the_delete(client, db_session):
    """Two identical DELETEs to one URL, differing only in the key.

    Run in this order on purpose: the owner's 204 afterwards proves the id was
    deletable all along, so the non-owner's 404 was the `client_name` filter
    and not a stale or wrong id.
    """
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    make_passage(db_session, acme_note, content=f"{ACME_MARKER} a passage.")
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    other_attempt = _delete_note(client, globex_key, acme_note.id)
    assert other_attempt.status_code == 404
    db_session.expire_all()
    assert (
        db_session.scalars(select(Note.id).where(Note.id == acme_note.id)).one_or_none()
        == acme_note.id
    )

    owner_attempt = _delete_note(client, acme_key, acme_note.id)
    assert owner_attempt.status_code == 204
    db_session.expire_all()
    assert (
        db_session.scalars(select(Note.id).where(Note.id == acme_note.id)).one_or_none()
        is None
    )


def test_delete_note_v2_404_is_identical_for_a_missing_and_a_foreign_note(
    client, db_session
):
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    globex_key = make_api_key(db_session, "globex")
    missing_id = acme_note.id + 1_000_000

    foreign = _delete_note(client, globex_key, acme_note.id)
    missing = _delete_note(client, globex_key, missing_id)

    assert foreign.status_code == missing.status_code == 404
    assert foreign.text == missing.text
    assert foreign.json() == {"detail": "Note not found"}


# ---------------------------------------------------------------------------
# POST /api/v2/search
# ---------------------------------------------------------------------------


def test_search_v2_does_not_return_another_clients_passages(client, db_session):
    acme_note = make_note(db_session, "acme", name="Acme Ledger")
    make_passage(db_session, acme_note, content=_body(ACME_MARKER))
    globex_key = make_api_key(db_session, "globex")

    response = _search(client, globex_key, q=_body(ACME_MARKER))

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert ACME_MARKER not in response.text


def test_the_client_filter_is_what_excludes_the_passage_in_v2(client, db_session):
    """The same body posted twice with different keys.

    Globex holds its own marker-bearing passage matching the same query, so
    the statement under test really runs for both callers -- a tenant with no
    rows would answer empty whether or not the filter were there.
    """
    acme_note = make_note(db_session, "acme", name="Acme Ledger")
    acme_passage = make_passage(db_session, acme_note, content=_body(ACME_MARKER))
    globex_note = make_note(db_session, "globex", name="Globex Ledger")
    globex_passage = make_passage(
        db_session, globex_note, content=_body(GLOBEX_MARKER)
    )
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    owner_view = _search(client, acme_key)
    other_view = _search(client, globex_key)

    assert [r["passage_id"] for r in owner_view.json()["results"]] == [
        acme_passage.id
    ]
    assert [r["passage_id"] for r in other_view.json()["results"]] == [
        globex_passage.id
    ]
    assert ACME_MARKER not in other_view.text
    assert GLOBEX_MARKER not in owner_view.text


def test_search_v2_repo_filter_does_not_cross_the_tenant_boundary(client, db_session):
    """repo_filter narrows within a tenant; it never widens across tenants."""
    acme_note = make_note(
        db_session, "acme", scope="shared", path_or_uri="docs/acme.md"
    )
    acme_passage = make_passage(db_session, acme_note, content=_body(ACME_MARKER))
    globex_note = make_note(
        db_session, "globex", scope="shared", path_or_uri="docs/globex.md"
    )
    globex_passage = make_passage(
        db_session, globex_note, content=_body(GLOBEX_MARKER)
    )
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")

    owner_view = _search(client, acme_key, repo_filter="shared")
    other_view = _search(client, globex_key, repo_filter="shared")

    assert [r["passage_id"] for r in owner_view.json()["results"]] == [
        acme_passage.id
    ]
    assert [r["passage_id"] for r in other_view.json()["results"]] == [
        globex_passage.id
    ]
    assert GLOBEX_MARKER not in owner_view.text
    assert ACME_MARKER not in other_view.text


def test_search_v2_a_colliding_note_id_under_two_tenants_does_not_leak(
    client, db_session
):
    """The join hazard: note metadata must come from the OWNER's note.

    The notes are seeded so the two tenants' ids interleave, which is the
    arrangement a join confused about ownership would get wrong. Nothing is
    hard-coded: the assertions run over the ids actually returned.

    If the top-level `Note.client_name == self.client_name` predicate were
    dropped and only the join condition remained, this is what catches it --
    the response would still be a 200 carrying another tenant's `name`,
    `description`, `path_or_uri`, `type` and provenance.
    """
    notes = []
    for index, owner in enumerate(("acme", "globex", "acme", "globex")):
        marker = ACME_MARKER if owner == "acme" else GLOBEX_MARKER
        note = make_note(
            db_session,
            owner,
            scope="shared",
            path_or_uri=f"docs/{owner}-{index}.md",
            name=f"{owner.title()} Ledger {index}",
            description=f"{marker} description of ledger {index}",
            type_=f"{owner}-type",
            provenance_kind="user override" if owner == "acme" else "empirical",
        )
        make_passage(db_session, note, content=_body(marker))
        notes.append(note)
    by_id = {note.id: note for note in notes}
    # The seeding order really did interleave the two tenants' ids.
    assert [by_id[note_id].client_name for note_id in sorted(by_id)] == [
        "acme",
        "globex",
        "acme",
        "globex",
    ]
    keys = {
        "acme": make_api_key(db_session, "acme"),
        "globex": make_api_key(db_session, "globex"),
    }

    for owner, other_marker in (
        ("acme", GLOBEX_MARKER),
        ("globex", ACME_MARKER),
    ):
        response = _search(client, keys[owner])
        results = response.json()["results"]
        assert len(results) == 2, f"{owner} must match both of its own passages"
        for result in results:
            note = by_id[result["note_id"]]
            assert note.client_name == owner
            # Every note-sourced field is the owner's own, field by field --
            # the join cannot have substituted the neighbouring id's row.
            assert result["name"] == note.name
            assert result["description"] == note.description
            assert result["path_or_uri"] == note.path_or_uri
            assert result["type"] == note.type
            assert result["provenance_kind"] == note.provenance_kind
            assert result["provenance_date"] == note.provenance_date
        assert other_marker not in response.text


def test_search_v2_does_not_return_a_passage_whose_note_belongs_to_another_tenant(
    client, db_session
):
    """This leak is unrepresentable, and the schema is why.

    A passage carries its own `client_name`, so in principle one could sit
    under another tenant's note and be returned to the wrong caller. No query
    prevents that; the composite FK (note_id, client_name) -> notes(id,
    client_name) does, by refusing the row at insert time. So the assertion is
    on the constraint, not on a search response: there is no such passage to
    search for.
    """
    acme_note = make_note(db_session, "acme", body=_body(ACME_MARKER))
    # Seeded before the failing flush: afterwards the session needs a rollback
    # and nothing else can be written.
    make_api_key(db_session, "globex")

    with pytest.raises(IntegrityError):
        make_passage(
            db_session,
            acme_note,
            content=f"{GLOBEX_MARKER} a passage claiming the wrong tenant.",
            client_name="globex",
        )

    db_session.rollback()
