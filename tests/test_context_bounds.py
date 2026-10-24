"""The two bounds on the context-expansion FETCH (issue #48).

tests/test_search_context.py covers what a region contains. This file covers
what the service is willing to read in order to build one, which is a separate
contract with its own failure mode: `MAX_CONTEXT_CHARS` on its own is spent on
text that has already crossed the wire, so a `context: "document"` search over
50 matched notes used to transfer every one of them -- embedding column
included -- and then discard most of it (README, "Both bounds apply before the
rows are read").

Four claims are asserted here that no response-shape assertion can reach:

- the context queries never select `embedding`. That is a claim about SQL, so
  it is asserted against the SQL the driver is handed, captured with a
  `before_cursor_execute` listener.
- what is NOT fetched. Asserted against the ranges
  `get_chunks_for_ranges` is called with, because a document that was measured
  as too large must cost zero bytes of text, not merely be absent from the
  response.
- both bounds are tenant-scoped. The measuring query is a second place
  `client_name` has to appear, and a leak there is silent: it would mis-price
  another tenant's rows and withhold context, which looks exactly like a tight
  budget.
- the budget is spent in RANK order, not in the order the ranges happen to be
  planned in (the regression fixed in 9f69c09). A lower-ranked region buying
  the budget first withholds a region the builder would have admitted, and the
  response still looks perfectly well-formed.

Rows are seeded through `make_learning` with hand-built embeddings rather than
through the ingest endpoint wherever ranking matters: a query of
`unit_vector(0)` against a row whose embedding is `s` on axis 0 and
`sqrt(1 - s^2)` on another axis has cosine similarity exactly `s`, so which
document is rank 1 is arithmetic the test controls rather than a property of
`fake_embedding`'s output. Region text is still asserted by string equality
against an independently computed slice of the original note (`span`), never a
length or a `startswith`.
"""
import math
import random

import pytest
from sqlalchemy import event

from app.api.v1 import search as search_module
from app.config import settings
from app.core.chunking import chunk_text
from app.core.context import (
    ContextHit,
    RangeSize,
    _region_chars_lower_bound,
    admit_ranges,
)
from app.repositories.learning import LearningRepository

from tests.factories import make_api_key, make_learning
from tests.fakes import unit_vector

# One definition of "a well-formed document" across the suite.
from tests.test_reassembly import OVERLAP, SIZE, STEP, document_of_length

REPO = "repo_a"

# The axis every "this row is not a hit" filler embedding points along. Any
# axis but 0 will do: it is orthogonal to the query, so the row's cosine
# similarity is exactly 0.0 and it ranks below every row seeded with a
# positive similarity.
_FILLER_AXIS = 1


def span(first: int, last: int, text: str) -> str:
    """The text chunks `first..last` of `text` cover, clipped at its end.

    Chunk j is `text[j*STEP : j*STEP + SIZE]`, so a run spans from the start of
    its first chunk to the end of its last. Computed from the original string
    rather than from the stored rows, so it is an independent expectation
    rather than a restatement of whatever the code produced. Deliberately a
    copy of the helper in tests/test_search_context.py: the arithmetic is the
    expectation, and an expectation shared with the module whose text it
    checks is worth less than a re-derived one.
    """
    return text[first * STEP : min(last * STEP + SIZE, len(text))]


def _note(tag: str, length: int) -> str:
    """A distinct, non-repeating note of exactly `length` characters."""
    return (f"NOTE-{tag} " + document_of_length(length + len(tag) + 7))[:length]


def _ranked_embedding(similarity: float, axis: int) -> list[float]:
    """A unit vector whose cosine similarity to `unit_vector(0)` is exactly
    `similarity`.

    `axis` must not be 0, and two rows meant to tie must share it: the
    component outside axis 0 carries no information the search reads, it only
    makes the vector a unit vector so that the dot product IS the cosine.
    """
    assert axis != 0, "axis 0 carries the similarity itself"
    vector = [0.0] * settings.embedding_dimension
    vector[0] = similarity
    vector[axis] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return vector


def _query_vector_is_axis_0(monkeypatch):
    """Make every query embed to `unit_vector(0)`, whatever its text.

    Patched on the route's module namespace, which is where the `client`
    fixture patches it too -- `search.py` did `from ... import get_embedding`,
    so patching `app.core.embeddings` would have no effect on what the
    endpoint calls.
    """
    monkeypatch.setattr(search_module, "get_embedding", lambda text: unit_vector(0))


def _seed_note(
    db,
    client_name: str,
    file_path: str,
    content: str,
    similarity_by_index: dict[int, float] | None = None,
    repo_name: str = REPO,
) -> list[str]:
    """Store `content` chunked exactly as ingest would, row by row.

    `make_learning` rather than the ingest endpoint because ranking has to be
    controlled: `similarity_by_index` gives the chunks named in it an
    embedding of that exact cosine similarity to `unit_vector(0)`, and every
    other chunk one of similarity 0.0. Passing None leaves `make_learning`'s
    default content-derived `fake_embedding`, for the tests where which row
    ranks first does not matter.

    No chunk is dropped, so the stored index sequence has no holes and the
    whole-document join is byte-exact (see tests/test_reassembly.py).
    """
    chunks = chunk_text(content, SIZE, OVERLAP)
    for index, chunk in enumerate(chunks):
        embedding = None
        if similarity_by_index is not None:
            embedding = _ranked_embedding(
                similarity_by_index.get(index, 0.0), _FILLER_AXIS
            )
        make_learning(
            db,
            client_name,
            repo_name=repo_name,
            file_path=file_path,
            content=chunk,
            chunk_index=index,
            embedding=embedding,
        )
    db.flush()
    db.expire_all()
    return chunks


def _search(client, key, query="anything", **payload):
    return client.post(
        "/api/v1/search",
        json={"query": query, **payload},
        headers={"X-API-Key": key},
    )


def _refs_by_path(body) -> dict[str, set]:
    refs: dict[str, set] = {}
    for result in body["results"]:
        refs.setdefault(result["metadata"]["relative_path"], set()).add(
            result["context_ref"]
        )
    return refs


# ---------------------------------------------------------------------------
# What the context queries ask the database for
# ---------------------------------------------------------------------------


class _SqlCapture:
    """The SQL each repository method handed the driver, per method.

    The listener alone cannot tell the three statements apart -- the search
    and the context fetch both select from `learnings` -- so each repository
    method is wrapped to label the statements executed inside it. Wrapping
    also means a missing method (as `measure_ranges` would be on the
    pre-#48 repository) fails loudly here rather than being read as "no
    embedding was selected".
    """

    def __init__(self):
        self.search: list[str] = []
        self.measure: list[str] = []
        self.fetch: list[str] = []
        self._stack: list[list[str]] = []

    def record(self, statement: str) -> None:
        if self._stack:
            self._stack[-1].append(statement)


@pytest.fixture
def captured_sql(connection, monkeypatch):
    capture = _SqlCapture()

    def before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        capture.record(statement)

    event.listen(connection, "before_cursor_execute", before_cursor_execute)

    for attribute, sink in (
        ("search", capture.search),
        ("measure_ranges", capture.measure),
        ("get_chunks_for_ranges", capture.fetch),
    ):
        original = getattr(LearningRepository, attribute)

        def labelled(self, *args, _original=original, _sink=sink, **kwargs):
            capture._stack.append(_sink)
            try:
                return _original(self, *args, **kwargs)
            finally:
                capture._stack.pop()

        monkeypatch.setattr(LearningRepository, attribute, labelled)

    yield capture

    event.remove(connection, "before_cursor_execute", before_cursor_execute)


@pytest.mark.parametrize("context", ["neighbors", "document"])
def test_neither_context_query_selects_the_embedding_column(
    client, db_session, captured_sql, context
):
    """#48: the 1024-dim embedding is ~33x the payload this path needs.

    Asserted against the SQL text rather than against timing or response
    size, because the regression -- `select(Learning, ...)`, which loads the
    whole mapped entity -- is invisible in the response: the route reads
    content and meta and never looks at the vector it paid to transfer.
    """
    key = make_api_key(db_session, "acme")
    note = _note("sql", 2600)
    chunks = _seed_note(db_session, "acme", "docs/sql.md", note)

    response = _search(
        client, key, chunks[1], top_k=5, context=context, context_chunks=1
    )
    assert response.status_code == 200, response.text
    assert response.json()["contexts"], "the fetch must actually have happened"

    context_statements = captured_sql.measure + captured_sql.fetch
    assert captured_sql.measure, "the measuring query must have run"
    assert captured_sql.fetch, "the fetch query must have run"
    for statement in context_statements:
        assert "embedding" not in statement.lower(), statement

    # The columns the fetch does need are there, so this is not passing by
    # failing to select anything.
    assert any("learnings.content" in statement for statement in captured_sql.fetch)
    assert any("learnings.meta" in statement for statement in captured_sql.fetch)

    # And the capture would see the column if it were selected -- the search's
    # own query, in the same transaction, does select it.
    assert any(
        "learnings.embedding" in statement for statement in captured_sql.search
    ), captured_sql.search


# ---------------------------------------------------------------------------
# Both bounds are tenant-scoped
# ---------------------------------------------------------------------------


def _seed_two_tenants_on_one_path(db_session):
    """The same repo_name/file_path owned by two clients, acme's note huge.

    The size asymmetry is the point: if the measuring query saw acme's rows,
    globex's own 2600-character note would be priced at over 46,000
    characters and withheld from a budget it comfortably fits.
    """
    path = "docs/shared.md"
    globex_note = _note("globex", 2600)
    acme_note = _note("acme", 44600)
    _seed_note(db_session, "globex", path, globex_note)
    _seed_note(db_session, "acme", path, acme_note)
    return path, globex_note, acme_note


def test_the_measuring_query_measures_only_the_callers_rows(db_session):
    path, globex_note, acme_note = _seed_two_tenants_on_one_path(db_session)
    globex_chunks = chunk_text(globex_note, SIZE, OVERLAP)

    rows = LearningRepository(db_session, client_name="globex").measure_ranges(
        [(REPO, path, None, None)]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.range_id == 0
    assert row.rows == len(globex_chunks)
    assert row.total_chars == sum(len(chunk) for chunk in globex_chunks)
    assert row.recorded_step == STEP
    # Stated as what acme's presence would have produced, so the assertions
    # above cannot pass by measuring nothing at all.
    assert row.rows < len(chunk_text(acme_note, SIZE, OVERLAP))


def test_the_context_fetch_returns_only_the_callers_rows(db_session):
    path, globex_note, acme_note = _seed_two_tenants_on_one_path(db_session)
    globex_chunks = chunk_text(globex_note, SIZE, OVERLAP)

    rows = LearningRepository(db_session, client_name="globex").get_chunks_for_ranges(
        [(REPO, path, None, None)]
    )

    assert [row.content for row in rows] == globex_chunks
    assert all("NOTE-acme" not in row.content for row in rows)


def test_a_tenants_region_is_unaffected_by_another_tenants_copy_of_the_path(
    client, db_session, monkeypatch
):
    """End to end: a leak in EITHER context query shows up here.

    A leak in the fetch splices acme's text into globex's region; a leak in
    the measurement prices globex's note at acme's size and withholds the
    region entirely. The budget is set tight enough to make the second
    outcome a failure rather than a silent pass: 3000 characters admits
    globex's note (measured at 1600) and would reject the pair (46,400).
    """
    path, globex_note, acme_note = _seed_two_tenants_on_one_path(db_session)
    globex_key = make_api_key(db_session, "globex")
    globex_chunks = chunk_text(globex_note, SIZE, OVERLAP)
    monkeypatch.setattr(settings, "max_context_chars", 3000)

    response = _search(client, globex_key, globex_chunks[0], top_k=50, context="document")

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["contexts"]) == 1
    region = body["contexts"][0]
    assert region["content"] == globex_note
    assert region["exact"] is True
    assert region["note"] is None, "nothing was bounded, so nothing is explained"
    assert all(result["context_ref"] == 0 for result in body["results"])
    assert len(body["results"]) == len(globex_chunks)
    # Nothing of acme's note reaches the wire at all, region or result.
    assert "NOTE-acme" not in response.text


# ---------------------------------------------------------------------------
# The document cap (MAX_CONTEXT_DOCUMENTS)
# ---------------------------------------------------------------------------


# Four single-chunk notes with hand-picked similarities, so the rank order of
# the documents is arithmetic rather than an accident of the fake embedder.
_CAP_SIMILARITIES = {"d1": 0.9, "d2": 0.8, "d3": 0.7, "d4": 0.6}


@pytest.fixture
def four_documents(client, db_session, monkeypatch):
    """One chunk per note, ranked d1 > d2 > d3 > d4, plus an acme key."""
    _query_vector_is_axis_0(monkeypatch)
    key = make_api_key(db_session, "acme")
    contents = {}
    for tag, similarity in _CAP_SIMILARITIES.items():
        contents[tag] = _note(tag, 300)
        make_learning(
            db_session,
            "acme",
            repo_name=REPO,
            file_path=f"docs/{tag}.md",
            content=contents[tag],
            chunk_index=0,
            embedding=_ranked_embedding(similarity, _FILLER_AXIS),
        )
    db_session.flush()
    db_session.expire_all()
    return key, contents


def test_the_document_cap_expands_only_the_best_ranked_documents(
    client, four_documents, monkeypatch
):
    """Two of four notes are expanded, and they are the top two by rank.

    The cap bounds the FETCH, so which notes it keeps has to be the ranking's
    answer and not the order the hits happened to be grouped in. The notes
    here are deliberately tiny -- the character budget is nowhere near spent,
    so the only thing that can withhold a region is the cap.
    """
    key, contents = four_documents
    monkeypatch.setattr(settings, "max_context_documents", 2)

    response = _search(client, key, top_k=50, context="document")

    assert response.status_code == 200, response.text
    body = response.json()
    assert [region["file_path"] for region in body["contexts"]] == [
        "docs/d1.md",
        "docs/d2.md",
    ]
    assert [region["content"] for region in body["contexts"]] == [
        contents["d1"],
        contents["d2"],
    ]
    # Every result survives -- the cap removes context, never results.
    assert len(body["results"]) == 4
    assert [result["context_ref"] for result in body["results"]] == [0, 1, None, None]
    assert _refs_by_path(body) == {
        "docs/d1.md": {0},
        "docs/d2.md": {1},
        "docs/d3.md": {None},
        "docs/d4.md": {None},
    }


def test_the_document_cap_explains_itself_on_the_last_region(
    client, four_documents, monkeypatch
):
    """The nulls above are explained in the only place adjacent to the cut.

    A caller holding `context_ref: null` otherwise cannot tell "that note
    could not be resolved" from "the service declined to read it", so the
    note names the setting and the number of documents it held back.
    """
    key, _ = four_documents
    monkeypatch.setattr(settings, "max_context_documents", 2)

    body = _search(client, key, top_k=50, context="document").json()

    assert body["contexts"][0]["note"] is None, "only the LAST region carries it"
    assert body["contexts"][-1]["note"] == (
        "2 lower-ranked document(s) were not expanded because the 2-document "
        "limit (MAX_CONTEXT_DOCUMENTS) was reached, and the results "
        "referencing them carry context_ref: null"
    )


@pytest.mark.parametrize("cap", [4, 5])
def test_a_cap_at_or_above_the_document_count_changes_nothing(
    client, four_documents, monkeypatch, cap
):
    """The control: at or above the number of notes matched, the cap is invisible.

    Compared against a run at the default of 25 rather than against a
    hand-written expectation, because "identical" is the claim -- not merely
    "four regions and no note".
    """
    key, _ = four_documents

    uncapped = _search(client, key, top_k=50, context="document").json()
    monkeypatch.setattr(settings, "max_context_documents", cap)
    capped = _search(client, key, top_k=50, context="document").json()

    assert capped == uncapped
    assert len(uncapped["contexts"]) == 4
    assert all(region["note"] is None for region in uncapped["contexts"])
    assert all(result["context_ref"] is not None for result in uncapped["results"])


# ---------------------------------------------------------------------------
# The pre-flight measurement (MAX_CONTEXT_CHARS, before the rows are read)
# ---------------------------------------------------------------------------


@pytest.fixture
def fetched_ranges(monkeypatch):
    """Every range `get_chunks_for_ranges` is actually asked for.

    What was NOT fetched is the whole claim of the pre-flight check, and it
    is unobservable in the response: a document that was measured as too
    large and a document that was fetched and then discarded by the region
    builder both come back as `context_ref: null`.
    """
    calls: list[list[tuple]] = []
    original = LearningRepository.get_chunks_for_ranges

    def spy(self, ranges):
        calls.append(list(ranges))
        return original(self, ranges)

    monkeypatch.setattr(LearningRepository, "get_chunks_for_ranges", spy)
    return calls


def test_a_document_measured_as_too_large_is_never_fetched(
    client, db_session, monkeypatch, fetched_ranges
):
    """The rank-1 note is 20,000 characters, so it is priced out and skipped.

    The two 2600-character notes behind it then keep their rank order and
    their exact text. Before the measurement existed the giant note was
    transferred in full, the region builder spent the whole budget rejecting
    it, and NOTHING came back expanded -- so this asserts both the ranges
    asked for and the regions produced.
    """
    _query_vector_is_axis_0(monkeypatch)
    key = make_api_key(db_session, "acme")
    giant = _note("giant", 20000)
    small_a = _note("smalla", 2600)
    small_b = _note("smallb", 2600)
    _seed_note(db_session, "acme", "docs/giant.md", giant, {i: 0.9 for i in range(25)})
    _seed_note(db_session, "acme", "docs/small_a.md", small_a, {i: 0.8 for i in range(4)})
    _seed_note(db_session, "acme", "docs/small_b.md", small_b, {i: 0.7 for i in range(4)})

    # 5200 admits both small notes (measured at 1600 each, really 2600 each)
    # and rejects the giant one (measured at 18,400).
    monkeypatch.setattr(settings, "max_context_chars", 5200)

    response = _search(client, key, top_k=50, context="document")

    assert response.status_code == 200, response.text
    body = response.json()

    # The claim about I/O: the giant note's rows were never asked for.
    assert len(fetched_ranges) == 1
    assert sorted(path for _, path, _, _ in fetched_ranges[0]) == [
        "docs/small_a.md",
        "docs/small_b.md",
    ]

    # The claim about the response: rank order kept, text exact.
    assert [region["file_path"] for region in body["contexts"]] == [
        "docs/small_a.md",
        "docs/small_b.md",
    ]
    assert [region["content"] for region in body["contexts"]] == [small_a, small_b]
    assert all(region["exact"] is True for region in body["contexts"])
    assert _refs_by_path(body) == {
        "docs/giant.md": {None},
        "docs/small_a.md": {0},
        "docs/small_b.md": {1},
    }
    # The giant note's matched chunks are still returned in full: the bound
    # withholds context, not results.
    assert len(body["results"]) == 25 + 4 + 4


def test_a_document_measured_as_too_large_says_so_before_fetching(
    client, db_session, monkeypatch
):
    """The pre-flight wording is distinct from the post-fetch one.

    "measured as too large to fit and were not expanded" is the only signal a
    caller gets that the service declined to read the note at all, as opposed
    to having read it and run out of response budget.
    """
    _query_vector_is_axis_0(monkeypatch)
    key = make_api_key(db_session, "acme")
    _seed_note(
        db_session, "acme", "docs/giant.md", _note("giant", 20000),
        {i: 0.9 for i in range(25)},
    )
    _seed_note(
        db_session, "acme", "docs/small_a.md", _note("smalla", 2600),
        {i: 0.8 for i in range(4)},
    )
    monkeypatch.setattr(settings, "max_context_chars", 5200)

    body = _search(client, key, top_k=50, context="document").json()

    assert len(body["contexts"]) == 1
    assert body["contexts"][-1]["note"] == (
        "the 5200-character context budget was reached before fetching: "
        "1 document(s) were measured as too large to fit and were not "
        "expanded, so the results referencing them carry context_ref: null"
    )


def test_the_post_fetch_budget_still_omits_what_the_estimate_admitted(
    client, db_session, monkeypatch
):
    """The region builder remains the authority on the budget.

    The estimate is deliberately a LOWER bound, so it admits documents the
    real text cannot fit: two 2600-character notes are measured at 1600 each,
    both are admitted into a 3400-character budget, and then the second one's
    real 2600 characters do not fit. That path -- and its own, distinct
    wording, with no document measured out -- has to keep working, because it
    is what makes the under-estimate safe.
    """
    _query_vector_is_axis_0(monkeypatch)
    key = make_api_key(db_session, "acme")
    first = _note("first", 2600)
    second = _note("second", 2600)
    _seed_note(db_session, "acme", "docs/first.md", first, {i: 0.9 for i in range(4)})
    _seed_note(db_session, "acme", "docs/second.md", second, {i: 0.8 for i in range(4)})
    monkeypatch.setattr(settings, "max_context_chars", 3400)

    body = _search(client, key, top_k=50, context="document").json()

    assert [region["file_path"] for region in body["contexts"]] == ["docs/first.md"]
    assert body["contexts"][0]["content"] == first
    assert body["contexts"][0]["note"] == (
        "the 3400-character context budget was reached after this region; "
        "1 further region(s) were omitted and the results referencing them "
        "carry context_ref: null"
    )
    assert _refs_by_path(body) == {"docs/first.md": {0}, "docs/second.md": {None}}


# ---------------------------------------------------------------------------
# The budget is spent in rank order (9f69c09)
# ---------------------------------------------------------------------------


# Chunk 52 has to exist for the rank-1 range's tail probe, so note A is long
# enough to carry 56 chunks; note B only needs its chunk 5.
_A_LENGTH = 44600
_B_LENGTH = 20600
_A_SIMILARITIES = {50: 0.9, 2: 0.5, 6: 0.4, 10: 0.3}
_B_SIMILARITIES = {3: 0.8}


def test_the_fetch_budget_is_spent_in_rank_order_not_in_planning_order(
    client, db_session, monkeypatch, fetched_ranges
):
    """The regression in #48's first fix: a document's own low-ranked ranges
    could buy the budget before a higher-ranked range in another document.

    Hits rank A@50, B@3, A@2, A@6, A@10. `plan_ranges` groups by document and
    sorts by index inside it, so it emits A[1..4], A[5..8], A[9..12],
    A[49..52], B[2..5] -- and a walk in THAT order spends a 5200-character
    budget on A's rank-3, rank-4 and rank-5 ranges, leaving nothing for A's
    rank-1 range or B's rank-2 one. The response still looks well-formed:
    two regions, exact text, a note explaining the budget. It is simply the
    wrong two regions.

    Spending the same budget in `_rank_key` order -- the order the region
    builder also ranks in -- buys A's rank-1 region and B's rank-2 one, which
    is what this asserts.
    """
    _query_vector_is_axis_0(monkeypatch)
    key = make_api_key(db_session, "acme")
    note_a = _note("alpha", _A_LENGTH)
    note_b = _note("bravo", _B_LENGTH)
    chunks_a = _seed_note(db_session, "acme", "docs/a.md", note_a, _A_SIMILARITIES)
    chunks_b = _seed_note(db_session, "acme", "docs/b.md", note_b, _B_SIMILARITIES)
    # The premises the arithmetic below rests on, asserted rather than assumed.
    assert len(chunks_a) == 56
    assert len(chunks_b) == 26

    monkeypatch.setattr(settings, "max_context_chars", 5200)

    response = _search(
        client, key, top_k=5, context="neighbors", context_chunks=1
    )

    assert response.status_code == 200, response.text
    body = response.json()

    # The ranking the rest of this test is about, stated as the hits it
    # produced: A@50, B@3, A@2, A@6, A@10.
    assert [
        (result["metadata"]["relative_path"], result["metadata"]["chunk_index"])
        for result in body["results"]
    ] == [
        ("docs/a.md", 50),
        ("docs/b.md", 3),
        ("docs/a.md", 2),
        ("docs/a.md", 6),
        ("docs/a.md", 10),
    ]

    # Exactly the rank-1 and rank-2 regions, in that order, with their text
    # independently computed from the original notes.
    assert [
        (
            region["file_path"],
            region["chunk_index_from"],
            region["chunk_index_to"],
        )
        for region in body["contexts"]
    ] == [("docs/a.md", 49, 51), ("docs/b.md", 2, 4)]
    assert body["contexts"][0]["content"] == span(49, 51, note_a)
    assert body["contexts"][1]["content"] == span(2, 4, note_b)
    assert all(region["exact"] is True for region in body["contexts"])
    assert [result["context_ref"] for result in body["results"]] == [
        0,
        1,
        None,
        None,
        None,
    ]

    # A's rank-1 range was bought, so it was fetched; the two ranges the
    # budget could not reach were not.
    assert len(fetched_ranges) == 1
    assert sorted(fetched_ranges[0]) == [
        ("repo_a", "docs/a.md", 1, 4),
        ("repo_a", "docs/a.md", 49, 52),
        ("repo_a", "docs/b.md", 2, 5),
    ]


def test_both_bounds_firing_quote_the_budget_once(
    client, db_session, monkeypatch
):
    """One budget, spent once, so the number appears once.

    The case above trips BOTH halves of the character budget: A[5..8] and
    A[9..12] are measured out before the fetch, and A[1..3] is built and then
    omitted after it. Two independent sentences would quote 5200 twice and
    read as two separate limits.
    """
    _query_vector_is_axis_0(monkeypatch)
    key = make_api_key(db_session, "acme")
    _seed_note(db_session, "acme", "docs/a.md", _note("alpha", _A_LENGTH), _A_SIMILARITIES)
    _seed_note(db_session, "acme", "docs/b.md", _note("bravo", _B_LENGTH), _B_SIMILARITIES)
    monkeypatch.setattr(settings, "max_context_chars", 5200)

    body = _search(client, key, top_k=5, context="neighbors", context_chunks=1).json()

    note = body["contexts"][-1]["note"]
    assert note.count("5200-character") == 1, note
    assert "context budget was reached" in note
    assert note == (
        "the 5200-character context budget was reached: 1 document(s) were "
        "measured as too large to fetch and 1 further region(s) were omitted "
        "after this one, so the results referencing them carry "
        "context_ref: null"
    )


# ---------------------------------------------------------------------------
# Generous bounds are invisible
# ---------------------------------------------------------------------------


def test_generous_bounds_leave_the_regions_exactly_as_they_were(
    client, db_session, monkeypatch
):
    """At the shipped defaults the bounds must not be observable at all.

    Both were added to a path that already worked, so the regression they
    could cause is a withheld region on an ordinary search. Asserted against
    independently computed spans and `note is None`, not against a previous
    run.
    """
    _query_vector_is_axis_0(monkeypatch)
    key = make_api_key(db_session, "acme")
    first = _note("one", 4400)
    second = _note("two", 4400)
    _seed_note(db_session, "acme", "docs/one.md", first, {1: 0.9})
    _seed_note(db_session, "acme", "docs/two.md", second, {3: 0.8})
    assert settings.max_context_chars == 100_000
    assert settings.max_context_documents == 25

    body = _search(
        client, key, top_k=2, context="neighbors", context_chunks=1
    ).json()

    assert [
        (region["file_path"], region["chunk_index_from"], region["chunk_index_to"])
        for region in body["contexts"]
    ] == [("docs/one.md", 0, 2), ("docs/two.md", 2, 4)]
    assert body["contexts"][0]["content"] == span(0, 2, first)
    assert body["contexts"][1]["content"] == span(2, 4, second)
    assert all(region["exact"] is True for region in body["contexts"])
    assert all(region["note"] is None for region in body["contexts"])
    assert [result["context_ref"] for result in body["results"]] == [0, 1]


# ---------------------------------------------------------------------------
# The property the pre-flight check rests on (no database)
# ---------------------------------------------------------------------------


def _sized(estimate: int) -> RangeSize:
    """A measurement whose lower-bound estimate is exactly `estimate`.

    `_region_chars_lower_bound` is `min(total_chars, (rows - 2) * step)`, so
    putting the number in `total_chars` and padding `rows` well past it makes
    the bound equal it. Asserted in the test rather than trusted.
    """
    return RangeSize(rows=estimate // STEP + 5, total_chars=estimate, recorded_step=None)


def _reference_rank_key(chunk_range, hits):
    """What rank a range holds, re-derived from the documented rule.

    Descending best similarity among the hits inside the range (the tail
    probe at `to_index` excluded, since no hit lives there), then the document
    key and the range's lowest index. Written out here rather than imported
    from app/core/context.py on purpose: a property checked against the
    implementation's own ordering would hold however that ordering changed.
    """
    repo_name, file_path, from_index, to_index = chunk_range
    similarities = [
        hit.similarity
        for hit in hits
        if hit.repo_name == repo_name
        and hit.file_path == file_path
        and hit.chunk_index is not None
        and (from_index is None or hit.chunk_index >= from_index)
        and (to_index is None or hit.chunk_index <= to_index - 1)
    ]
    assert similarities, "every generated range carries at least one hit"
    return (
        -max(similarities),
        repo_name,
        file_path,
        0 if from_index is None else from_index,
    )


def _greedy_with_real_lengths(ranges, real_lengths, budget, hits):
    """The regions the builder would admit, as a set of range positions.

    The region builder walks candidates in rank order and STOPS at the first
    one that does not fit, because what it returns has to be a rank-ordered
    prefix. This mirrors that walk with the real lengths, which is the
    behaviour the pre-flight check must never withhold a range from.
    """
    admitted = set()
    spent = 0
    for position in sorted(
        range(len(ranges)), key=lambda p: _reference_rank_key(ranges[p], hits)
    ):
        length = real_lengths[position]
        if spent + length > budget:
            break
        spent += length
        admitted.add(position)
    return admitted


def _random_case(rng: random.Random):
    """One randomised (ranges, sizes, hits, real_lengths, budget) case.

    Ranges within a document are kept disjoint and each carries at least one
    hit, which is what `plan_ranges` guarantees its output: it merges
    overlapping and adjacent spans and derives every span from a hit's index.
    Every estimate is drawn at or below the range's real length -- the
    premise the whole argument rests on.
    """
    ranges = []
    hits = []
    sizes = {}
    real_lengths = {}
    for document in range(rng.randint(1, 3)):
        repo_name = f"repo_{document}"
        file_path = f"docs/{document}.md"
        next_index = 0
        for _ in range(rng.randint(1, 3)):
            from_index = next_index + rng.randint(0, 5)
            to_index = from_index + rng.randint(1, 6)
            next_index = to_index + 2
            position = len(ranges)
            ranges.append((repo_name, file_path, from_index, to_index))
            for _ in range(rng.randint(1, 2)):
                hits.append(
                    ContextHit(
                        repo_name=repo_name,
                        file_path=file_path,
                        chunk_index=rng.randint(from_index, to_index - 1),
                        similarity=round(rng.uniform(0.0, 1.0), 6),
                    )
                )
            real_lengths[position] = rng.randrange(0, 6000)
            # A position with no measurement at all -- a range that matched no
            # rows -- costs nothing and is admitted, so leave some out.
            if rng.random() < 0.9:
                sizes[position] = _sized(rng.randint(0, real_lengths[position]))

    # Hits reach admit_ranges in rank order, as the route passes them.
    hits.sort(key=lambda hit: -hit.similarity)
    budget = rng.randrange(0, sum(real_lengths.values()) + 2000)
    return ranges, sizes, hits, real_lengths, budget


def test_admit_ranges_never_withholds_a_range_the_builder_would_use(monkeypatch):
    """The safety property of the pre-flight check, over randomised cases.

    Measuring before fetching is only sound if it cannot withhold context
    that would have fitted. Since each estimate is a lower bound on what its
    range's regions really contribute, and both walks use the same rank
    order, every range the region builder would have admitted must have been
    admitted here too. Seeded, so a failure is reproducible.
    """
    rng = random.Random(20261024)
    for case in range(400):
        ranges, sizes, hits, real_lengths, budget = _random_case(rng)
        monkeypatch.setattr(settings, "max_context_chars", budget)

        # The harness itself: each estimate really is the value _sized meant.
        for position, size in sizes.items():
            assert _region_chars_lower_bound(size) <= real_lengths[position]

        admitted, skipped_documents = admit_ranges(ranges, sizes, hits)

        needed = _greedy_with_real_lengths(ranges, real_lengths, budget, hits)
        admitted_positions = {
            position
            for position, chunk_range in enumerate(ranges)
            if chunk_range in admitted
        }
        assert needed <= admitted_positions, (
            f"case {case}: the builder would have used {sorted(needed)} but "
            f"only {sorted(admitted_positions)} were fetched "
            f"(budget {budget})"
        )

        # The returned order is the input's, and the skipped count is the
        # number of distinct documents that lost a range.
        assert admitted == [
            chunk_range
            for position, chunk_range in enumerate(ranges)
            if position in admitted_positions
        ]
        assert skipped_documents == len(
            {
                (chunk_range[0], chunk_range[1])
                for position, chunk_range in enumerate(ranges)
                if position not in admitted_positions
            }
        )
