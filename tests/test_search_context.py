"""Context expansion in POST /api/v1/search, end to end.

A chunk is a 1000-character positional window with 200 characters of overlap,
so a hit routinely begins and ends mid-thought. Context expansion gives the
caller what surrounded it -- without ever becoming a way to fetch a note you
could not find by asking for it (see
docs/superpowers/specs/2026-10-03-search-context-expansion-design.md).

Two properties get the most attention here, because they are the ones that fail
quietly:

- An expanded region must be the REAL surrounding text. Concatenating chunks
  duplicates the shared overlap at every seam, and the result still looks like
  prose -- so every region assertion below is string equality against an
  independently computed slice of the originally ingested note, never a length
  or a `startswith`.
- Expansion must never fail the search. The matched chunks are the answer to
  the caller's question and are already in hand; context is strictly additive.
  Every degradation path therefore asserts a 200 WITH its results intact, not
  merely that no exception escaped.

`tests/fakes.fake_embedding` is content-derived -- the same text always yields
the same vector -- so querying with a chunk's exact text puts that chunk at
cosine distance 0 and makes which hit ranks first deterministic. Several tests
below depend on that to isolate a single hit.
"""
import pytest

from app.api.v1 import search as search_module
from app.config import settings
from app.core.chunking import chunk_text
from app.models.learning import Learning
from app.repositories.learning import LearningRepository
from sqlalchemy import select

from tests.factories import make_api_key, make_learning

# One definition of "a well-formed document" across the suite.
from tests.test_reassembly import OVERLAP, SIZE, STEP, document_of_length

REPO = "repo_a"
PATH = "docs/note.md"

# 5000 characters at the production window is seven chunks: six full-size ones
# and a 200-character tail. Long enough that an interior region has full-size
# chunks on both sides of it, which is what makes the mid-document exactness
# test below meaningful.
NOTE = document_of_length(5000)
NOTE_CHUNKS = chunk_text(NOTE, SIZE, OVERLAP)
EXPECTED_CHUNKS = 7


def span(first: int, last: int, text: str = NOTE) -> str:
    """The text chunks `first..last` of `text` cover, clipped at its end.

    Chunk j is `text[j*STEP : j*STEP + SIZE]`, so a run spans from the start of
    its first chunk to the end of its last. Computed from the original string
    rather than from the stored chunks, so it is an independent expectation
    rather than a restatement of whatever the code produced.
    """
    return text[first * STEP : min(last * STEP + SIZE, len(text))]


def _ingest(client, key, content, repo_name=REPO, file_path=PATH):
    response = client.post(
        "/api/v1/ingest",
        data={"repo_name": repo_name, "file_path": file_path, "content": content},
        headers={"X-API-Key": key},
    )
    assert response.status_code == 200, response.text
    return response.json()["ingested"]


def _search(client, key, query, **payload):
    return client.post(
        "/api/v1/search",
        json={"query": query, **payload},
        headers={"X-API-Key": key},
    )


def _regions_by_span(body):
    return {
        (region["chunk_index_from"], region["chunk_index_to"]): region
        for region in body["contexts"]
    }


@pytest.fixture
def seeded(client, db_session):
    """An acme key with NOTE ingested through the real endpoint."""
    key = make_api_key(db_session, "acme")
    assert _ingest(client, key, NOTE) == EXPECTED_CHUNKS
    db_session.expire_all()
    return key


# ---------------------------------------------------------------------------
# The default: no context, and no second query
# ---------------------------------------------------------------------------


def test_the_default_mode_adds_no_context(client, seeded):
    """`context` is omitted entirely, so this pins the DEFAULT, not "none".

    Every existing caller sends no `context` field, and must be unaffected.
    """
    response = _search(client, seeded, NOTE_CHUNKS[2], top_k=5)

    assert response.status_code == 200
    body = response.json()
    assert body["results"], "the search itself must still return hits"
    assert body["contexts"] == []
    assert all(result["context_ref"] is None for result in body["results"])


def test_the_default_mode_runs_no_context_query(client, seeded, monkeypatch):
    """"No extra query" is a claim about I/O, so it is asserted against I/O.

    `contexts == []` alone would also hold if the range query ran and returned
    nothing useful. Spying on the repository method is what actually proves the
    round trip is skipped.
    """
    calls = []
    original = LearningRepository.get_chunks_for_ranges

    def spy(self, ranges):
        calls.append(ranges)
        return original(self, ranges)

    monkeypatch.setattr(LearningRepository, "get_chunks_for_ranges", spy)

    assert _search(client, seeded, NOTE_CHUNKS[2], top_k=5).status_code == 200
    assert calls == []

    # And the spy works -- otherwise the assertion above proves nothing.
    assert (
        _search(client, seeded, NOTE_CHUNKS[2], top_k=1, context="document").status_code
        == 200
    )
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# What a region contains
# ---------------------------------------------------------------------------


def test_neighbors_returns_the_hit_plus_one_chunk_either_side(client, seeded):
    """The region is the exact slice of the note that chunks 1..3 cover.

    `top_k=1` isolates a single hit on purpose: with a larger top_k every chunk
    of this note ranks, their neighbour windows all merge, and the region
    becomes the whole note -- which is a different test (see the merging one
    below).
    """
    response = _search(
        client, seeded, NOTE_CHUNKS[2], top_k=1, context="neighbors", context_chunks=1
    )

    body = response.json()
    assert [result["content"] for result in body["results"]] == [NOTE_CHUNKS[2]]
    assert len(body["contexts"]) == 1
    region = body["contexts"][0]
    assert (region["chunk_index_from"], region["chunk_index_to"]) == (1, 3)
    # Overlap stripped: chunks 1..3 are 3000 characters of stored text and the
    # span they cover is 2600, so a concatenating implementation fails here.
    assert region["content"] == span(1, 3)
    assert len(region["content"]) == 2600
    assert region["covers"] == "neighbors"
    assert region["exact"] is True
    assert region["note"] is None
    assert body["results"][0]["context_ref"] == 0


def test_document_returns_the_whole_note_byte_identical(client, seeded):
    response = _search(client, seeded, NOTE_CHUNKS[2], top_k=1, context="document")

    body = response.json()
    assert len(body["contexts"]) == 1
    region = body["contexts"][0]
    assert region["content"] == NOTE
    assert (region["chunk_index_from"], region["chunk_index_to"]) == (
        0,
        EXPECTED_CHUNKS - 1,
    )
    assert region["covers"] == "document"
    assert region["exact"] is True
    assert region["note"] is None


@pytest.mark.parametrize("context_chunks", [1, 2, 3])
def test_a_neighbors_region_widens_with_context_chunks(client, seeded, context_chunks):
    """The width is the parameter's, and the text still matches the slice."""
    response = _search(
        client,
        seeded,
        NOTE_CHUNKS[3],
        top_k=1,
        context="neighbors",
        context_chunks=context_chunks,
    )

    region = response.json()["contexts"][0]
    expected_from = 3 - context_chunks
    expected_to = min(3 + context_chunks, EXPECTED_CHUNKS - 1)
    assert (region["chunk_index_from"], region["chunk_index_to"]) == (
        expected_from,
        expected_to,
    )
    assert region["content"] == span(expected_from, expected_to)


def test_a_mid_document_region_is_exact_despite_a_full_size_last_chunk(client, seeded):
    """The route-level mirror of the `is_document_tail` unit test.

    Tail completeness says a run ending on a full-size chunk lost a successor.
    That is true at the end of a note and false in the middle, where the
    successor is stored and merely outside the region. Chunk 3 of this note is
    full size, so without the distinction every interior neighbours region
    would come back `exact: false` with a loss reported that did not happen.
    """
    # The premise, asserted rather than assumed.
    assert len(NOTE_CHUNKS[3]) == SIZE
    assert 3 < EXPECTED_CHUNKS - 1, "chunk 3 must not be the note's tail"

    response = _search(
        client, seeded, NOTE_CHUNKS[2], top_k=1, context="neighbors", context_chunks=1
    )

    region = response.json()["contexts"][0]
    assert region["chunk_index_to"] == 3
    assert region["exact"] is True
    assert region["note"] is None
    assert region["content"] == span(1, 3)


# ---------------------------------------------------------------------------
# Merging and deduplication
# ---------------------------------------------------------------------------


def test_hits_across_one_note_share_a_single_merged_region(client, seeded):
    """Seven overlapping neighbour windows become one region, text once.

    With `top_k=50` every chunk of the note ranks, so the planned spans are
    [0,1], [0,2], [1,3] ... [5,6] -- all overlapping, and all one region. The
    point of the sidecar is that the text appears once no matter how many
    results point at it.
    """
    response = _search(
        client, seeded, NOTE_CHUNKS[0], top_k=50, context="neighbors", context_chunks=1
    )

    body = response.json()
    assert len(body["results"]) == EXPECTED_CHUNKS
    assert len(body["contexts"]) == 1, "overlapping windows must merge into one region"
    assert all(result["context_ref"] == 0 for result in body["results"])
    # Deduplicated: one copy of the note, not one per result. Counted against
    # the raw response body, because that is where the cost of duplication
    # would actually land -- seven copies of this note would be a response
    # seven times larger, spending the model's context window on the same
    # passage repeatedly. A 400-character probe is used rather than the whole
    # note because the matched chunks themselves legitimately repeat the text.
    assert body["contexts"][0]["content"] == NOTE
    assert response.text.count(NOTE[2000:2400]) == 2  # once in a chunk, once in the region


def test_two_hits_in_one_note_under_document_share_one_region(client, seeded):
    response = _search(client, seeded, NOTE_CHUNKS[1], top_k=2, context="document")

    body = response.json()
    assert len(body["results"]) == 2
    assert len(body["contexts"]) == 1
    assert [result["context_ref"] for result in body["results"]] == [0, 0]
    assert body["contexts"][0]["content"] == NOTE


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------


def _delete_chunk(db_session, index, file_path=PATH):
    rows = db_session.scalars(
        select(Learning).where(Learning.file_path == file_path)
    ).all()
    target = next(row for row in rows if row.meta["chunk_index"] == index)
    db_session.delete(target)
    db_session.flush()
    db_session.expire_all()


def test_a_hole_splits_the_region_rather_than_splicing(client, db_session, seeded):
    """A dropped chunk's characters are in no neighbour, so never splice.

    Joining across the hole would make chunks 2 and 4 look adjacent when 800
    characters of the note sit between them. The region is cut instead, and
    the two sides become separate regions with their own refs.
    """
    _delete_chunk(db_session, 3)

    response = _search(client, seeded, NOTE_CHUNKS[0], top_k=50, context="document")

    body = response.json()
    assert response.status_code == 200
    regions = _regions_by_span(body)
    assert set(regions) == {(0, 2), (4, 6)}

    # Each side is the exact slice it covers -- and critically, neither
    # contains any part of the other.
    assert regions[(0, 2)]["content"] == span(0, 2)
    assert regions[(4, 6)]["content"] == span(4, 6)
    assert span(4, 6) not in regions[(0, 2)]["content"]

    # The caller is told why each region stops where it does.
    assert "truncated above chunk 2" in regions[(0, 2)]["note"]
    assert "truncated below chunk 4" in regions[(4, 6)]["note"]

    # A hit on each side of the hole points at its OWN region, not a shared one.
    refs_by_index = {
        result["metadata"]["chunk_index"]: result["context_ref"]
        for result in body["results"]
    }
    assert refs_by_index[1] is not None
    assert refs_by_index[5] is not None
    assert refs_by_index[1] != refs_by_index[5]
    assert 3 not in refs_by_index, "the deleted chunk must not come back as a hit"


def test_a_hole_does_not_make_a_neighbors_region_splice(client, db_session, seeded):
    """The same rule under `neighbors`, where the span reaches over the hole."""
    _delete_chunk(db_session, 3)

    response = _search(
        client, seeded, NOTE_CHUNKS[2], top_k=1, context="neighbors", context_chunks=2
    )

    body = response.json()
    assert len(body["contexts"]) == 1
    region = body["contexts"][0]
    # Asked for 0..4; chunk 3 is gone, so the region stops at 2.
    assert (region["chunk_index_from"], region["chunk_index_to"]) == (0, 2)
    assert region["content"] == span(0, 2)
    assert "truncated above chunk 2" in region["note"]


# ---------------------------------------------------------------------------
# The character budget
# ---------------------------------------------------------------------------


def _tagged_note(tag: str, length: int = 2500) -> str:
    """A distinct note of exactly `length` characters, carrying `tag`."""
    return (f"NOTE-{tag} " + document_of_length(length))[:length]


def test_the_budget_omits_lower_ranked_regions_and_says_so(
    client, db_session, monkeypatch
):
    """Regions are built in rank order until the budget is spent.

    Three notes of 2500 characters and a 2600-character budget: exactly one
    region fits. The one that fits must be the highest-ranked -- the note
    containing the exact-match chunk -- and the results whose regions were
    dropped must say so through a null ref rather than silently losing context.
    """
    key = make_api_key(db_session, "acme")
    notes = {tag: _tagged_note(tag) for tag in ("a", "b", "c")}
    for tag, note in notes.items():
        _ingest(client, key, note, file_path=f"docs/{tag}.md")
    db_session.expire_all()

    monkeypatch.setattr(settings, "max_context_chars", 2600)

    # Query with a chunk of note "b", so "b" is the top-ranked region.
    query = chunk_text(notes["b"], SIZE, OVERLAP)[0]
    response = _search(client, key, query, top_k=50, context="document")

    body = response.json()
    assert response.status_code == 200
    assert len(body["contexts"]) == 1, "only one 2500-char region fits in 2600"
    region = body["contexts"][0]
    assert region["file_path"] == "docs/b.md"
    assert region["content"] == notes["b"]
    # The cut is explained on the last region that did fit -- the only place
    # adjacent to it, since an omitted region is absent by contract.
    assert "context budget was reached" in region["note"]
    assert "context_ref: null" in region["note"]

    # Results in note b point at it; results in the omitted notes carry null.
    refs_by_path = {}
    for result in body["results"]:
        refs_by_path.setdefault(result["metadata"]["relative_path"], set()).add(
            result["context_ref"]
        )
    assert refs_by_path["docs/b.md"] == {0}
    assert refs_by_path["docs/a.md"] == {None}
    assert refs_by_path["docs/c.md"] == {None}

    # The matched chunks themselves are untouched by the budget -- it caps
    # context, not results.
    assert len(body["results"]) == sum(
        len(chunk_text(note, SIZE, OVERLAP)) for note in notes.values()
    )


def test_a_budget_that_fits_everything_omits_nothing(client, db_session):
    """The control for the test above: no note, no nulls, at the default budget."""
    key = make_api_key(db_session, "acme")
    notes = {tag: _tagged_note(tag) for tag in ("a", "b", "c")}
    for tag, note in notes.items():
        _ingest(client, key, note, file_path=f"docs/{tag}.md")
    db_session.expire_all()

    body = _search(
        client,
        key,
        chunk_text(notes["b"], SIZE, OVERLAP)[0],
        top_k=50,
        context="document",
    ).json()

    assert len(body["contexts"]) == 3
    assert all(region["note"] is None for region in body["contexts"])
    assert all(result["context_ref"] is not None for result in body["results"])
    assert {region["content"] for region in body["contexts"]} == set(notes.values())


# ---------------------------------------------------------------------------
# Degradation: never a non-200
# ---------------------------------------------------------------------------


def test_a_note_whose_rows_disagree_on_the_window_yields_no_region(
    client, db_session
):
    """One unresolvable note costs its own context and nothing else.

    Rows of one note always come from a single ingest run, so disagreeing
    recorded windows mean they were written outside the application and the
    window cannot be trusted. Joining under a guessed window silently
    duplicates the overlap at every seam, so the region is dropped -- but the
    search, the matched chunks, and every OTHER note's region survive.
    """
    key = make_api_key(db_session, "acme")
    _ingest(client, key, NOTE, file_path="docs/broken.md")
    good = _tagged_note("good")
    _ingest(client, key, good, file_path="docs/good.md")

    rows = db_session.scalars(
        select(Learning).where(Learning.file_path == "docs/broken.md")
    ).all()
    # Reassignment, not in-place mutation: plain JSONB has no change tracking.
    rows[1].meta = {**rows[1].meta, "chunk_overlap": OVERLAP + 100}
    db_session.flush()
    db_session.expire_all()

    response = _search(client, key, NOTE_CHUNKS[2], top_k=50, context="document")

    body = response.json()
    assert response.status_code == 200
    # No region for the broken note; the healthy one is unaffected.
    assert [region["file_path"] for region in body["contexts"]] == ["docs/good.md"]
    assert body["contexts"][0]["content"] == good

    refs_by_path = {}
    for result in body["results"]:
        refs_by_path.setdefault(result["metadata"]["relative_path"], set()).add(
            result["context_ref"]
        )
    assert refs_by_path["docs/broken.md"] == {None}
    assert refs_by_path["docs/good.md"] == {0}

    # The broken note's matched chunks are still returned in full -- expansion
    # is additive and can only ever fail to add.
    returned = [
        result["content"]
        for result in body["results"]
        if result["metadata"]["relative_path"] == "docs/broken.md"
    ]
    assert sorted(returned) == sorted(NOTE_CHUNKS)


def test_an_unexpected_failure_in_expansion_still_returns_the_search(
    client, seeded, monkeypatch
):
    """The route's blanket `except Exception` is load-bearing, so it is tested.

    Anything unanticipated inside expansion -- a malformed metadata shape, a
    database error on the extra query, a bug in region building -- must cost
    the context and not the search. A narrower catch would mean the failures
    nobody predicted are exactly the ones that return a 500 on a search whose
    answer was already in hand.
    """

    def explode(*args, **kwargs):
        raise RuntimeError("deliberate failure inside context expansion")

    monkeypatch.setattr(search_module, "build_regions", explode)

    response = _search(client, seeded, NOTE_CHUNKS[2], top_k=5, context="document")

    assert response.status_code == 200, response.text
    body = response.json()
    # The primary answer survives intact, with real content.
    assert len(body["results"]) == 5
    assert NOTE_CHUNKS[2] in [result["content"] for result in body["results"]]
    assert body["contexts"] == []
    assert all(result["context_ref"] is None for result in body["results"])


def test_a_failure_in_the_context_query_still_returns_the_search(
    client, seeded, monkeypatch
):
    """The same guarantee when the extra QUERY is what fails, not the builder."""

    def explode(self, ranges):
        raise RuntimeError("deliberate database failure on the context query")

    monkeypatch.setattr(LearningRepository, "get_chunks_for_ranges", explode)

    response = _search(
        client, seeded, NOTE_CHUNKS[2], top_k=3, context="neighbors", context_chunks=2
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["results"]) == 3
    assert body["contexts"] == []
    assert all(result["context_ref"] is None for result in body["results"])


def test_a_hit_whose_chunk_index_is_malformed_loses_only_its_context(
    client, db_session
):
    """A row that cannot be placed in the index sequence is not expandable.

    It still matches and is still returned -- it just gets no region, which is
    this feature's preferred failure everywhere.
    """
    key = make_api_key(db_session, "acme")
    content = "a chunk whose recorded index is not an integer at all, so unplaceable"
    row = make_learning(
        db_session, "acme", repo_name=REPO, file_path="docs/odd.md", content=content
    )
    row.meta = {**row.meta, "chunk_index": "second"}
    db_session.flush()
    db_session.expire_all()

    response = _search(client, key, content, top_k=5, context="document")

    assert response.status_code == 200
    body = response.json()
    assert [result["content"] for result in body["results"]] == [content]
    assert body["results"][0]["context_ref"] is None
    assert body["contexts"] == []


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("context_chunks", [1, 5])
def test_the_declared_context_chunks_bounds_are_accepted(
    client, seeded, context_chunks
):
    response = _search(
        client,
        seeded,
        NOTE_CHUNKS[2],
        top_k=1,
        context="neighbors",
        context_chunks=context_chunks,
    )

    assert response.status_code == 200
    assert response.json()["contexts"], "an accepted bound must still expand"


@pytest.mark.parametrize("context_chunks", [0, 6, -1])
def test_context_chunks_outside_its_bounds_is_rejected(
    client, seeded, context_chunks
):
    response = _search(
        client,
        seeded,
        NOTE_CHUNKS[2],
        context="neighbors",
        context_chunks=context_chunks,
    )

    assert response.status_code == 422


def test_an_unknown_context_mode_is_rejected(client, seeded):
    assert (
        _search(client, seeded, NOTE_CHUNKS[2], context="everything").status_code == 422
    )


@pytest.mark.parametrize("context", ["none", "neighbors", "document"])
def test_every_declared_context_mode_is_accepted(client, seeded, context):
    assert _search(client, seeded, NOTE_CHUNKS[2], context=context).status_code == 200
