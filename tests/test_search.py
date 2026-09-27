"""Search behaviour against real pgvector.

Ranking assertions use basis vectors so the cosine distances are exact rather
than approximate: distance to itself is 0.0, distance to an orthogonal vector
is 1.0, and similarity is reported as 1 - distance.
"""
import pytest

from app.api.v1 import search as search_module
from tests.factories import make_api_key, make_learning
from tests.fakes import unit_vector

# pytest, search_module and unit_vector are unused until Task 8 appends the
# ranking tests to this file. They are declared here so that task does not have
# to reopen the import block. Do not remove them as "unused".


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
