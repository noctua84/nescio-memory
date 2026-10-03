"""The tenant boundary.

Every failure mode here is silent in production: no exception, no error
response, just one client reading or destroying another client's data.
"""
from sqlalchemy import select

from app.core.chunking import chunk_text
from app.models.learning import Learning
from tests.factories import make_api_key, make_learning

LONG_ENOUGH = (
    "This content comfortably exceeds the fifty character minimum that the "
    "ingest endpoint applies to each chunk."
)


def test_search_does_not_return_another_clients_learnings(client, db_session):
    make_learning(db_session, "acme", content="a private acme note")
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/search",
        json={"query": "a private acme note", "top_k": 50},
        headers={"X-API-Key": globex_key},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_a_client_sees_only_its_own_rows_when_both_exist(client, db_session):
    make_learning(db_session, "acme", content="acme flavoured content")
    make_learning(db_session, "globex", content="globex flavoured content")
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/search",
        json={"query": "flavoured content", "top_k": 50},
        headers={"X-API-Key": globex_key},
    )

    contents = [result["content"] for result in response.json()["results"]]
    assert contents == ["globex flavoured content"]


def test_reingest_by_another_client_does_not_delete_existing_rows(client, db_session):
    # delete_by_file filters on client_name. Without that filter, this ingest
    # would wipe acme's copy of the same path.
    make_learning(
        db_session,
        "acme",
        repo_name="shared",
        file_path="docs/note.md",
        content="acme content for the shared path",
    )
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/ingest",
        data={
            "repo_name": "shared",
            "file_path": "docs/note.md",
            "content": LONG_ENOUGH,
        },
        headers={"X-API-Key": globex_key},
    )
    assert response.status_code == 200

    db_session.expire_all()
    acme_rows = db_session.scalars(
        select(Learning).where(Learning.client_name == "acme")
    ).all()
    assert len(acme_rows) == 1
    assert acme_rows[0].content == "acme content for the shared path"


def test_the_same_path_coexists_under_two_clients(client, db_session):
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    payload = {
        "repo_name": "shared",
        "file_path": "docs/note.md",
        "content": LONG_ENOUGH,
    }

    for key in (acme_key, globex_key):
        assert (
            client.post(
                "/api/v1/ingest", data=payload, headers={"X-API-Key": key}
            ).status_code
            == 200
        )

    db_session.expire_all()
    owners = db_session.scalars(
        select(Learning.client_name).where(Learning.file_path == "docs/note.md")
    ).all()
    assert sorted(set(owners)) == ["acme", "globex"]


def test_the_client_filter_is_what_excludes_the_row(client, db_session):
    """Two requests differing only in the key presented; one sees the row.

    This is the teeth of the isolation suite. Because the requests are
    otherwise identical, a passing pair cannot be explained by anything except
    the client_name filter, so no mutation of production code is needed to show
    that the filter is load-bearing.
    """
    make_learning(db_session, "acme", content="the contested row")
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    payload = {"query": "the contested row", "top_k": 50}

    owner_view = client.post(
        "/api/v1/search", json=payload, headers={"X-API-Key": acme_key}
    )
    other_view = client.post(
        "/api/v1/search", json=payload, headers={"X-API-Key": globex_key}
    )

    assert [r["content"] for r in owner_view.json()["results"]] == [
        "the contested row"
    ]
    assert other_view.json()["results"] == []


def test_ingest_stamps_the_authenticated_clients_name(client, db_session):
    # LearningRepository.add stamps client_name; the endpoint never passes it.
    key = make_api_key(db_session, "acme")

    client.post(
        "/api/v1/ingest",
        data={
            "repo_name": "repo_a",
            "file_path": "docs/note.md",
            "content": LONG_ENOUGH,
        },
        headers={"X-API-Key": key},
    )

    db_session.expire_all()
    owners = db_session.scalars(select(Learning.client_name)).all()
    assert owners
    assert set(owners) == {"acme"}


# ---------------------------------------------------------------------------
# Context expansion in POST /api/v1/search
#
# A leaking search hands over the chunks that happen to rank. A leaking context
# expansion hands over whole notes -- and it does so through a SECOND query,
# get_chunks_for_ranges, which is reached with the (repo_name, file_path) of a
# hit. That path is not authorisation: it has to filter on client_name
# independently, because a filter that is correct only because an earlier
# filter was correct is one refactor away from being wrong. See
# docs/superpowers/specs/2026-10-03-search-context-expansion-design.md, D6.
# ---------------------------------------------------------------------------

# Multi-chunk, so expansion has real joining to do, and each carrying a marker
# that cannot appear in the other tenant's note -- the markers are what let a
# leak be asserted against the whole response body rather than one field.
ACME_MARKER = "ACME-ONLY-SENTINEL-7f3a"
GLOBEX_MARKER = "GLOBEX-ONLY-SENTINEL-91cd"


def _tenant_note(marker: str) -> str:
    """A ~2400-character note stamped with `marker` throughout.

    Repeated rather than placed once so the marker lands in every chunk: a
    leak of any single chunk is then detectable, not only a leak of the first.
    """
    sentence = (
        f"{marker} this note belongs to exactly one tenant and its contents "
        f"must never surface for another. {marker} "
    )
    return (sentence * 20)[:2400].strip()


def _search(client, key, **payload):
    return client.post(
        "/api/v1/search",
        json={"query": payload.pop("query"), **payload},
        headers={"X-API-Key": key},
    )


def test_two_clients_expand_their_own_note_from_the_same_path(client, db_session):
    """The leak that matters most for this feature: same repo, same path.

    Nothing in either request distinguishes the two notes -- both tenants hold
    `shared/docs/note.md` -- so the only thing deciding which text is expanded
    is the client_name filter on the context query. Asserted in both
    directions, because a response that returned the SAME note to both tenants
    would satisfy a one-sided check.
    """
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    acme_note = _tenant_note(ACME_MARKER)
    globex_note = _tenant_note(GLOBEX_MARKER)

    for key, note in ((acme_key, acme_note), (globex_key, globex_note)):
        assert (
            client.post(
                "/api/v1/ingest",
                data={
                    "repo_name": "shared",
                    "file_path": "docs/note.md",
                    "content": note,
                },
                headers={"X-API-Key": key},
            ).status_code
            == 200
        )

    db_session.expire_all()
    acme_view = _search(
        client, acme_key, query=acme_note[:200], top_k=50, context="document"
    )
    globex_view = _search(
        client, globex_key, query=globex_note[:200], top_k=50, context="document"
    )

    # Each tenant's expansion is its own note, byte for byte -- not a merge of
    # the two, and not the other's.
    assert [c["content"] for c in acme_view.json()["contexts"]] == [acme_note]
    assert [c["content"] for c in globex_view.json()["contexts"]] == [globex_note]
    # And no trace of the other tenant anywhere in the body: results,
    # contexts, metadata or otherwise.
    assert GLOBEX_MARKER not in acme_view.text
    assert ACME_MARKER not in globex_view.text


def test_the_client_filter_is_what_withholds_the_expanded_region(client, db_session):
    """Two requests differing only in the key presented; one gets the note.

    The same teeth as test_the_client_filter_is_what_excludes_the_row, applied
    to the context query. Because the requests are otherwise identical, what
    comes back cannot be explained by anything except the client_name filter on
    get_chunks_for_ranges.

    Globex deliberately holds ONE row at the contested path, rather than
    nothing. A tenant with no rows at all cannot detect a missing filter here:
    its search returns no hits, so no ranges are planned and the context query
    is never issued -- the request would pass against a completely unscoped
    expansion. Giving globex a hit at the same path is what forces the query to
    run and makes the assertion load-bearing. The no-rows case is still covered
    below, as the weaker half it is.
    """
    acme_note = _tenant_note(ACME_MARKER)
    for index, chunk in enumerate(chunk_text(acme_note)):
        make_learning(
            db_session,
            "acme",
            repo_name="shared",
            file_path="docs/contested.md",
            content=chunk,
            chunk_index=index,
        )
    globex_chunk = f"{GLOBEX_MARKER} globex's single row at the contested path."
    make_learning(
        db_session,
        "globex",
        repo_name="shared",
        file_path="docs/contested.md",
        content=globex_chunk,
        chunk_index=0,
    )
    db_session.flush()
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    initech_key = make_api_key(db_session, "initech")
    payload = {"query": acme_note[:200], "top_k": 50, "context": "document"}

    owner_view = _search(client, acme_key, **payload)
    other_view = _search(client, globex_key, **payload)
    empty_view = _search(client, initech_key, **payload)

    owner = owner_view.json()
    assert [c["content"] for c in owner["contexts"]] == [acme_note]
    assert owner["results"], "the owner must actually have matched something"
    assert all(result["context_ref"] == 0 for result in owner["results"])

    # Globex's expansion of the SAME path is its own single row and nothing
    # else -- not acme's note, and not the two spliced together.
    other = other_view.json()
    assert [c["content"] for c in other["contexts"]] == [globex_chunk]
    assert [r["content"] for r in other["results"]] == [globex_chunk]
    assert ACME_MARKER not in other_view.text

    # And a tenant holding nothing gets nothing: no results, so no region and
    # no ref to point with.
    empty = empty_view.json()
    assert empty["results"] == []
    assert empty["contexts"] == []
    assert ACME_MARKER not in empty_view.text


def test_neighbors_does_not_reach_across_a_tenant_boundary(client, db_session):
    """The collision the context query exists to get right.

    Both tenants hold `shared/docs/note.md`, and their chunk indices occupy
    the same space: acme has 0, 1 and 2, globex has only 1. Expanding globex's
    hit by one chunk either side asks for indices 0..2 of that path -- which
    for acme exist and for globex do not.

    Without the tenant filter on get_chunks_for_ranges, acme's chunks 0 and 2
    are fetched, pass the contiguity check (the indices really are 0,1,2), and
    are joined into globex's region with acme's text on both sides of its own.
    That is a silent leak inside a 200 response, which is why the filter sits
    at the top level of the WHERE clause rather than inside each disjunct.
    """
    acme_chunks = chunk_text(_tenant_note(ACME_MARKER))
    for index in (0, 1, 2):
        make_learning(
            db_session,
            "acme",
            repo_name="shared",
            file_path="docs/note.md",
            content=acme_chunks[index],
            chunk_index=index,
        )
    globex_chunk = (
        f"{GLOBEX_MARKER} globex holds exactly one chunk of this path, at the "
        f"index acme also uses. {GLOBEX_MARKER}"
    )
    make_learning(
        db_session,
        "globex",
        repo_name="shared",
        file_path="docs/note.md",
        content=globex_chunk,
        chunk_index=1,
    )
    db_session.flush()
    globex_key = make_api_key(db_session, "globex")

    response = _search(
        client,
        globex_key,
        query=globex_chunk,
        top_k=1,
        context="neighbors",
        context_chunks=1,
    )

    body = response.json()
    assert response.status_code == 200
    assert [result["content"] for result in body["results"]] == [globex_chunk]
    # One region, covering index 1 only -- globex's own chunk, alone.
    assert len(body["contexts"]) == 1
    region = body["contexts"][0]
    assert region["content"] == globex_chunk
    assert (region["chunk_index_from"], region["chunk_index_to"]) == (1, 1)
    # The assertion that fails if the tenant filter is dropped.
    assert ACME_MARKER not in response.text
