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
# name ("the #42 self-distance tests"). Everything in that module's zero-norm
# guard and in LearningRepository.search's `distance != NaN` filter rests on
# one empirical claim about the real extension: a vector whose float4 norm is
# zero has an UNDEFINED cosine distance, reported as NaN, even against itself.
# That claim cannot be derived from Python -- float64 arithmetic on the same
# values gives a perfectly ordinary nonzero norm -- so it is measured here,
# against pgvector/pgvector:pg17 in the container, rather than assumed. If a
# future pgvector returned 0 for these instead, both guards would become dead
# code and these tests are what would say so.

# Subnormal in float4 (below FLOAT32_MIN_NORMAL, 1.17549435e-38) and a
# perfectly ordinary nonzero float64.
SUBNORMAL = 1e-40

# Normal in float4, but 1e-25 squared is 1e-50, under float32's smallest
# subnormal -- so the norm underflows even though no single component does.
UNDERFLOWING_NORMAL = 1e-25

ZERO_NORM_STORED_VECTORS = [
    pytest.param([0.0] * settings.embedding_dimension, id="all-zero"),
    pytest.param([SUBNORMAL] * settings.embedding_dimension, id="all-subnormal"),
    pytest.param(
        [UNDERFLOWING_NORMAL] * settings.embedding_dimension,
        id="underflowing-norm",
    ),
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
    # The core premise. `v <=> v` is the friendliest question that can be
    # asked of a vector -- identical operands, so any norm that is nonzero
    # gives exactly 0 -- and these three answer NaN, because cosine distance
    # divides by a norm that float4 evaluates as zero. Asserted as a float
    # that is NaN rather than merely "not 0": a NULL or a raised error would
    # be a different defect with different handling, and `!= 0` would admit
    # both.
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
    # It is the STORED side that poisons the result, which is why the filter in
    # LearningRepository.search has to exclude rows rather than reject queries.
    # Each query here is a vector _validated() accepts, and the distance comes
    # back NaN regardless. Both operand orders, since `<=>` divides by the
    # product of the two norms and a one-sided guard would look sufficient from
    # whichever side happened to be tested.
    stored = [0.0] * settings.embedding_dimension
    assert math.isnan(_cosine_distance(connection, query, stored))
    assert math.isnan(_cosine_distance(connection, stored, query))


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
# search() carries its own guard, a `distance != NaN` filter. These tests seed
# such a row directly through the factory, which bypasses _validated() exactly
# as a pre-existing row does.
#
# The vector seeded is the all-zero one specifically, because that is the one
# the SQL filter can see: NaN arises when the dot product is zero as well as
# the norm, which an all-zero stored vector guarantees against every query
# (the parametrised premise test above pins exactly that). The subnormal and
# underflowing-norm vectors are rejected on the way IN by _validated(), which
# is where their coverage lives (tests/test_embeddings.py).


def _zero_norm_row(db_session, content: str = "the zero-norm row", **kwargs):
    return make_learning(
        db_session,
        "acme",
        content=content,
        embedding=[0.0] * settings.embedding_dimension,
        **kwargs,
    )


def test_a_stored_zero_norm_row_is_left_out_of_the_results(
    client, db_session, query_on_axis
):
    # The defect, end to end: before the filter, this request returned HTTP
    # 200 with `"similarity": null` for the degenerate row, violating
    # SearchResult.similarity: float on a response the caller had no way to
    # recognise as broken. The row is simply absent now, and the healthy row
    # beside it is still returned -- an exclusion that also dropped the good
    # row would satisfy a test that only looked for the absence.
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        file_path="healthy.md",
        content="the healthy row",
        embedding=unit_vector(0),
    )
    _zero_norm_row(db_session, file_path="degenerate.md")
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["content"] for result in results] == ["the healthy row"]


def test_every_similarity_in_a_response_is_a_finite_float(
    client, db_session, query_on_axis
):
    # The schema's promise, asserted as the caller experiences it rather than
    # as a row count. `similarity: float` is declared in SearchResult, and a
    # NaN distance reaches JSON as `null` -- which pydantic serialises without
    # complaint on the way out, so the response validates and is still wrong.
    # isinstance plus math.isfinite is what distinguishes that: None fails the
    # first, and a future serialiser emitting NaN would fail the second.
    key = make_api_key(db_session, "acme")
    _zero_norm_row(db_session, file_path="degenerate.md")
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


def test_the_repository_itself_excludes_the_zero_norm_row(db_session):
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
    _zero_norm_row(db_session, file_path="degenerate.md")

    rows = LearningRepository(db_session, "acme").search(unit_vector(0), top_k=50)

    assert [learning.content for learning, _ in rows] == ["the healthy row"]
    # The similarity column is computed as 1 - distance, so a NaN that
    # survived the filter would arrive here rather than at the serialiser.
    for _, similarity in rows:
        assert math.isfinite(similarity)
