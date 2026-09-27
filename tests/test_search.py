"""Search behaviour against real pgvector.

Ranking assertions use basis vectors so the cosine distances are exact rather
than approximate: distance to itself is 0.0, distance to an orthogonal vector
is 1.0, and similarity is reported as 1 - distance.
"""
import pytest

from app.api.v1 import search as search_module
from tests.factories import make_api_key, make_learning
from tests.fakes import unit_vector


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
