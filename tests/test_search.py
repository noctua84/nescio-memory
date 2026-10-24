"""Search behaviour against real pgvector.

Ranking assertions use basis vectors so the cosine distances are exact rather
than approximate: distance to itself is 0.0, distance to an orthogonal vector
is 1.0, and similarity is reported as 1 - distance.
"""
import math

import pytest
from sqlalchemy import text

from app.api.v1 import search as search_module
from app.config import settings
from app.repositories.learning import LearningRepository
from tests.factories import make_api_key, make_learning
from tests.fakes import fake_embedding, unit_vector


def test_search_returns_a_stored_chunk(client, db_session):
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        repo_name="repo_a",
        file_path="docs/note.md",
        content="the chunk we expect to get back",
    )

    response = client.post(
        "/api/v1/search",
        json={"query": "the chunk we expect to get back", "top_k": 5},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["content"] == "the chunk we expect to get back"
    assert results[0]["metadata"]["relative_path"] == "docs/note.md"


@pytest.fixture
def query_on_axis(monkeypatch):
    """Pin the query embedding to a basis vector.

    The client fixture already patched get_embedding to the deterministic fake;
    this narrows it further so the query vector is exactly known. Re-patching
    works because the endpoint resolves the module global at call time.
    """

    def _pin(axis: int) -> None:
        monkeypatch.setattr(
            search_module, "get_embedding", lambda text: unit_vector(axis)
        )

    return _pin


def test_results_are_ordered_by_cosine_distance(client, db_session, query_on_axis):
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        file_path="near.md",
        content="on axis zero",
        embedding=unit_vector(0),
    )
    make_learning(
        db_session,
        "acme",
        file_path="far.md",
        content="on axis one",
        embedding=unit_vector(1),
    )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "irrelevant, the vector is pinned", "top_k": 5},
        headers={"X-API-Key": key},
    )

    results = response.json()["results"]
    assert [result["content"] for result in results] == [
        "on axis zero",
        "on axis one",
    ]
    # Exact, not approximate: identical vectors are distance 0, orthogonal
    # vectors are distance 1, and similarity is reported as 1 - distance.
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[1]["similarity"] == pytest.approx(0.0)


def test_top_k_limits_the_number_of_results(client, db_session, query_on_axis):
    key = make_api_key(db_session, "acme")
    for index in range(4):
        make_learning(
            db_session,
            "acme",
            file_path=f"chunk_{index}.md",
            content=f"chunk number {index}",
            embedding=unit_vector(index),
        )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 2},
        headers={"X-API-Key": key},
    )

    assert len(response.json()["results"]) == 2


def test_repo_filter_restricts_results_to_one_repository(
    client, db_session, query_on_axis
):
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        repo_name="repo_a",
        file_path="a.md",
        content="lives in repo a",
        embedding=unit_vector(0),
    )
    make_learning(
        db_session,
        "acme",
        repo_name="repo_b",
        file_path="b.md",
        content="lives in repo b",
        embedding=unit_vector(0),
    )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50, "repo_filter": "repo_a"},
        headers={"X-API-Key": key},
    )

    contents = [result["content"] for result in response.json()["results"]]
    assert contents == ["lives in repo a"]


def test_top_k_above_the_schema_maximum_is_rejected(client, db_session):
    # SearchRequest declares top_k with le=50.
    key = make_api_key(db_session, "acme")

    response = client.post(
        "/api/v1/search",
        json={"query": "anything", "top_k": 51},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 422


# --- #42 self-distance: what pgvector really does with a zero-norm vector ---
#
# These are the premise tests app/core/embeddings.py's comments point at by
# name ("the #42 self-distance tests"), and they are also what picks between
# the two filters in LearningRepository.search. Both rest on claims about the
# real extension that cannot be derived from Python -- float64 arithmetic on
# these same values gives a perfectly ordinary nonzero norm -- so they are
# measured against pgvector/pgvector:pg17 in the container rather than
# assumed:
#
#   1. a vector whose float4 norm is zero has an UNDEFINED cosine distance,
#      reported as NaN, even against ITSELF. That self-distance is pgvector's
#      own definition of "this stored vector has zero norm", and it is what
#      search()'s `self_distance != NaN` clause keys off.
#   2. against an ordinary QUERY vector, only the all-zero shape stays NaN.
#      The others produce an infinite similarity, which pgvector clamps into
#      [-1, 1] -- so their query-time distance lands at a perfectly finite
#      0.0 or 2.0 and a query-time `distance != NaN` clause cannot see them.
#
# Claim 2 is the reason the self-distance clause exists at all. If a future
# pgvector stopped clamping, or returned 0 for claim 1, these tests are what
# would say so rather than leaving a filter quietly matching nothing.

# Subnormal in float4 (below FLOAT32_MIN_NORMAL, 1.17549435e-38) and a
# perfectly ordinary nonzero float64.
SUBNORMAL = 1e-40

# Normal in float4, but 1e-25 squared is 1e-50, under float32's smallest
# subnormal -- so the norm underflows even though no single component does.
UNDERFLOWING_NORMAL = 1e-25

# Zero on the odd indices and subnormal on the even ones, starting at index 0.
# Mixed rather than uniform so the norm really is accumulated across both
# kinds: a guard that recognised a degenerate vector by its first component,
# or by its components all being equal, would pass the uniform cases and fail
# this one. Index 0 carries the nonzero value deliberately -- a unit vector on
# axis 0 then has a NONZERO dot product with it, which is what puts this case
# on the clamped side of claim 2 rather than the NaN side.
MIXED_ZERO_AND_SUBNORMAL = [
    SUBNORMAL if position % 2 == 0 else 0.0
    for position in range(settings.embedding_dimension)
]

ALL_ZERO = [0.0] * settings.embedding_dimension

# The degenerate shapes whose float4 norm is zero while their float64 norm is
# not. These are what _validated() refuses on the way in; a row carrying one
# predates that check, or was written by some other path.
NONZERO_ZERO_NORM_VECTORS = [
    pytest.param([SUBNORMAL] * settings.embedding_dimension, id="all-subnormal"),
    pytest.param(
        [UNDERFLOWING_NORMAL] * settings.embedding_dimension,
        id="underflowing-norm",
    ),
    pytest.param(MIXED_ZERO_AND_SUBNORMAL, id="mixed-zero-and-subnormal"),
]

# Every stored shape search() has to exclude: the all-zero one, which is NaN
# at query time too, and the three above, which are not.
ZERO_NORM_STORED_VECTORS = [
    pytest.param(ALL_ZERO, id="all-zero"),
    *NONZERO_ZERO_NORM_VECTORS,
]


def _vector_literal(vector: list[float]) -> str:
    """`vector` in pgvector's text input format.

    Passed as a bound parameter and cast in SQL rather than interpolated, so
    the value reaching float4 is the driver's own rendering of these numbers
    and not something this test rewrote on the way.
    """
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


def _cosine_distance(connection, left: list[float], right: list[float]) -> float:
    """`left <=> right` as pgvector computes it, with nothing in between.

    CAST(... AS vector) rather than `::vector`: SQLAlchemy's text() treats a
    colon as a bind-parameter marker, so the operator spelling would have to
    be escaped and the escaping, not the operator, would be what the test
    depended on.
    """
    return connection.execute(
        text("SELECT CAST(:left AS vector) <=> CAST(:right AS vector)"),
        {"left": _vector_literal(left), "right": _vector_literal(right)},
    ).scalar_one()


@pytest.mark.parametrize("vector", ZERO_NORM_STORED_VECTORS)
def test_42_self_distance_of_a_zero_norm_vector_is_nan(connection, vector):
    # Claim 1, and the clause it justifies: `v <=> v` is the friendliest
    # question that can be asked of a vector -- identical operands, so any
    # nonzero norm gives exactly 0 -- and every one of these answers NaN,
    # because cosine distance divides by a norm that float4 evaluates as zero.
    # That is why search() can ask the question of the STORED column alone,
    # without reference to the query. Asserted as a float that is NaN rather
    # than merely "not 0": a NULL or a raised error would be a different
    # defect with different handling, and `!= 0` would admit both.
    distance = _cosine_distance(connection, vector, vector)
    assert isinstance(distance, float)
    assert math.isnan(distance)


# Ordinary query vectors, each with a norm of exactly 1 or comfortably above
# it, so nothing about THEM could be responsible for an undefined distance.
ORDINARY_QUERY_VECTORS = [
    pytest.param(unit_vector(0), id="unit-vector"),
    pytest.param([0.01] * settings.embedding_dimension, id="uniform-positive"),
    pytest.param([-0.01] * settings.embedding_dimension, id="uniform-negative"),
    pytest.param(fake_embedding("an ordinary query"), id="fake-embedder-output"),
]


@pytest.mark.parametrize("query", ORDINARY_QUERY_VECTORS)
def test_42_a_stored_all_zero_vector_is_nan_against_any_query(connection, query):
    # Claim 2's exception, and the whole reason a query-time `distance != NaN`
    # clause looked sufficient for as long as it did. It is the STORED side
    # that poisons the result -- each query here is a vector _validated()
    # accepts -- and for the all-zero shape the dot product is zero as well as
    # the norm, so the division is 0/0 and stays NaN whatever is searched for.
    # Both operand orders, since `<=>` divides by the product of the two norms
    # and a one-sided guard would look sufficient from whichever side happened
    # to be tested.
    assert math.isnan(_cosine_distance(connection, query, ALL_ZERO))
    assert math.isnan(_cosine_distance(connection, ALL_ZERO, query))


@pytest.mark.parametrize("vector", NONZERO_ZERO_NORM_VECTORS)
def test_42_a_stored_zero_norm_vector_is_not_nan_against_an_ordinary_query(
    connection, vector
):
    """Claim 2: why the self-distance clause is not redundant.

    These three shapes have a zero float4 norm -- the self-distance test above
    says so -- and yet their distance against an ordinary query is NOT NaN.
    The dot product is nonzero while the norm is zero, so the similarity is
    +-Inf, and pgvector clamps similarity into [-1, 1] before returning
    `1 - similarity`. The degenerate row therefore arrives with a perfectly
    finite distance: 0.0 for a query it has a positive dot product with, which
    reads as a PERFECT MATCH, and 2.0 for a negative one.

    This is measured rather than reasoned about, and it is the premise the
    clause rests on: a query-time `distance != NaN` filter cannot see any of
    these rows, so without the stored-row self-distance clause they would be
    returned and ranked first. Pinned exactly (0.0 and 2.0, not merely "not
    NaN") because the clamping is what produces those two values; a pgvector
    that stopped clamping would hand back +-Inf, which is a different
    serialisation problem and should fail here rather than pass quietly.
    """
    positive = [0.01] * settings.embedding_dimension
    negative = [-0.01] * settings.embedding_dimension

    assert _cosine_distance(connection, positive, vector) == 0.0
    assert _cosine_distance(connection, negative, vector) == 2.0
    # And a unit vector on axis 0, which is what the search tests below query
    # with: every one of these shapes carries a nonzero component at index 0,
    # so the dot product is nonzero and the result is the clamped 0.0 -- the
    # degenerate row looking like an exact hit.
    assert _cosine_distance(connection, unit_vector(0), vector) == 0.0


def test_42_self_distance_of_an_ordinary_vector_is_zero(connection):
    # The control, without which every assertion above would be satisfied by a
    # pgvector that returned NaN for everything -- and the `!= NaN` filter in
    # search() would then be excluding every row in the table while the
    # premise tests stayed green. Exact rather than approximate: identical
    # operands of nonzero norm are distance 0, and orthogonal unit vectors are
    # distance 1, which is the arithmetic the ranking tests above rely on.
    assert _cosine_distance(connection, unit_vector(0), unit_vector(0)) == 0.0
    assert _cosine_distance(connection, unit_vector(0), unit_vector(1)) == 1.0


# --- #42: a stored zero-norm row must not reach a caller ---
#
# _validated() refuses to let one in through ingest, but rows predating that
# check -- or written by any other path -- are already in the table, and
# float4 can flatten a vector that looked fine in float64 on the way in. So
# search() carries its own guard on the stored row's self-distance. These
# tests seed such a row directly through the factory, which bypasses
# _validated() exactly as a pre-existing row does, and they cover every
# degenerate shape: the all-zero one, which is NaN at query time too, and the
# three that are not and therefore need the self-distance clause.


def _zero_norm_row(
    db_session,
    content: str = "the zero-norm row",
    embedding: list[float] | None = None,
    **kwargs,
):
    return make_learning(
        db_session,
        "acme",
        content=content,
        embedding=ALL_ZERO if embedding is None else embedding,
        **kwargs,
    )


@pytest.mark.parametrize("vector", ZERO_NORM_STORED_VECTORS)
def test_a_stored_zero_norm_row_is_left_out_of_the_results(
    client, db_session, query_on_axis, vector
):
    # The defect, end to end, in both of the forms it took. For the all-zero
    # shape the request returned HTTP 200 with `"similarity": null`, violating
    # SearchResult.similarity: float; for the other three it returned a
    # perfectly well-typed `1.0` for a row carrying no information at all.
    # Neither was something the caller could recognise as broken. The row is
    # simply absent now, and the healthy row beside it is still returned -- an
    # exclusion that also dropped the good row would satisfy a test that only
    # looked for the absence.
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        file_path="healthy.md",
        content="the healthy row",
        embedding=unit_vector(0),
    )
    _zero_norm_row(db_session, file_path="degenerate.md", embedding=vector)
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["content"] for result in results] == ["the healthy row"]


@pytest.mark.parametrize("vector", ZERO_NORM_STORED_VECTORS)
def test_every_similarity_in_a_response_is_a_finite_float(
    client, db_session, query_on_axis, vector
):
    # The schema's promise, asserted as the caller experiences it rather than
    # as a row count. `similarity: float` is declared in SearchResult, and a
    # NaN distance reaches JSON as `null` -- which pydantic serialises without
    # complaint on the way out, so the response validates and is still wrong.
    # isinstance plus math.isfinite is what distinguishes that: None fails the
    # first, and a future serialiser emitting NaN would fail the second.
    #
    # The row count is asserted alongside it because finiteness alone is not
    # the whole contract: three of these four shapes come back with a finite,
    # clamped 0.0 distance, so a response containing the degenerate row would
    # satisfy every isinstance/isfinite check above while still reporting a
    # row that means nothing.
    key = make_api_key(db_session, "acme")
    _zero_norm_row(db_session, file_path="degenerate.md", embedding=vector)
    for index in range(3):
        make_learning(
            db_session,
            "acme",
            file_path=f"healthy_{index}.md",
            content=f"healthy row {index}",
            embedding=unit_vector(index),
        )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50},
        headers={"X-API-Key": key},
    )

    results = response.json()["results"]
    assert len(results) == 3
    for result in results:
        assert isinstance(result["similarity"], float)
        assert math.isfinite(result["similarity"])


def test_the_zero_norm_filter_does_not_disturb_ranking_or_top_k(
    client, db_session, query_on_axis
):
    # The filter is a post-index WHERE clause and must stay one: it may remove
    # rows, and it may not reorder what is left or change how many are
    # returned. Both are asserted at once because a filter implemented as a
    # CASE in the ORDER BY, or one folded into the LIMIT, would still exclude
    # the degenerate row while quietly breaking one of these.
    key = make_api_key(db_session, "acme")
    _zero_norm_row(db_session, file_path="degenerate.md")
    for index in range(3):
        make_learning(
            db_session,
            "acme",
            file_path=f"axis_{index}.md",
            content=f"on axis {index}",
            embedding=unit_vector(index),
        )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 2},
        headers={"X-API-Key": key},
    )

    results = response.json()["results"]
    # top_k counts the rows the caller gets, so the excluded row must not be
    # spending one of the two slots.
    assert len(results) == 2
    assert results[0]["content"] == "on axis 0"
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[1]["similarity"] == pytest.approx(0.0)


def test_a_zero_norm_row_is_the_only_row_and_the_search_still_succeeds(
    client, db_session, query_on_axis
):
    # The degenerate shape of the exclusion: filtering out the only row must
    # produce an empty 200, not an error and not a row with a null score. A
    # caller's "no memories matched" path is ordinary; a 500 is not.
    key = make_api_key(db_session, "acme")
    _zero_norm_row(db_session)
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize("vector", ZERO_NORM_STORED_VECTORS)
def test_the_repository_itself_excludes_the_zero_norm_row(db_session, vector):
    """The guard at the layer that owns it, without the route in the way.

    The endpoint tests above would also pass if the row were dropped by the
    response mapping rather than by the query -- and a row filtered in Python
    still costs a LIMIT slot, so the top_k assertion would then be pinning
    luck. Asserting on search()'s own rows puts the exclusion where the
    comment in LearningRepository.search says it is: in the SQL.
    """
    make_learning(
        db_session,
        "acme",
        file_path="healthy.md",
        content="the healthy row",
        embedding=unit_vector(0),
    )
    _zero_norm_row(db_session, file_path="degenerate.md", embedding=vector)

    rows = LearningRepository(db_session, "acme").search(unit_vector(0), top_k=50)

    assert [learning.content for learning, _ in rows] == ["the healthy row"]
    # The similarity column is computed as 1 - distance, so a NaN that
    # survived the filter would arrive here rather than at the serialiser.
    for _, similarity in rows:
        assert math.isfinite(similarity)


# A unit vector at 45 degrees to axis 0: cosine similarity 1/sqrt(2) against
# unit_vector(0), so its distance is ~0.293 rather than 0. A genuinely
# relevant hit, and deliberately not a perfect one -- against a stored row
# that is also at distance 0 the comparison would be a tie and the ordering
# assertion below would be pinning luck rather than the defect.
_HALF_ROOT_TWO = 2 ** -0.5
OFF_AXIS_UNIT_VECTOR = [
    _HALF_ROOT_TWO if position in (0, 1) else 0.0
    for position in range(settings.embedding_dimension)
]


@pytest.mark.parametrize("vector", NONZERO_ZERO_NORM_VECTORS)
def test_a_stored_zero_norm_row_does_not_outrank_a_genuinely_relevant_one(
    client, db_session, query_on_axis, vector
):
    """The harm the self-distance clause actually prevents, stated as ranking.

    This is the case a query-time `distance != NaN` filter could not see at
    all. These stored shapes have a zero float4 norm and a NONZERO dot product
    with the query, so pgvector's similarity is +Inf, clamped to 1, and their
    distance arrives as 0.0 -- strictly better than the ~0.293 of the row that
    genuinely answers the query. The degenerate row therefore came back FIRST,
    with `similarity: 1.0`, pushing the real answer to second place: a
    well-typed, confidently-scored, entirely meaningless top hit. At top_k=1
    the real answer would not have been returned at all.

    Asserted as position and score rather than as mere absence, because that
    is the damage: the absence tests above would also pass for an
    implementation that excluded the row by luck of ordering, and a 200 whose
    first result is noise is worse for a caller than an error.
    """
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        file_path="relevant.md",
        content="the genuinely relevant row",
        embedding=OFF_AXIS_UNIT_VECTOR,
    )
    _zero_norm_row(db_session, file_path="degenerate.md", embedding=vector)
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50},
        headers={"X-API-Key": key},
    )

    results = response.json()["results"]
    assert [result["content"] for result in results] == ["the genuinely relevant row"]
    assert results[0]["similarity"] == pytest.approx(_HALF_ROOT_TWO)
    # And at top_k=1 -- where a degenerate row ranked first costs the caller
    # the real answer outright rather than merely demoting it.
    narrow = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 1},
        headers={"X-API-Key": key},
    )
    narrow_results = narrow.json()["results"]
    assert [result["content"] for result in narrow_results] == [
        "the genuinely relevant row"
    ]
