"""The tenant boundary for /api/v2 -- ingest and get.

Delete and search are the other half, in
tests/test_isolation_v2_delete_search.py; the suite was split to keep each
module under the length the plan asks for.

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
from sqlalchemy import select

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
