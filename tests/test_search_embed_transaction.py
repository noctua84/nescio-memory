"""The auth read must not hold a transaction open while the query is embedded.

get_current_client runs a SELECT on the request's session, which autobegins a
transaction and checks a pool connection out. The query embed can then wait up
to QUERY_EMBED_TIMEOUT_S on Ollama, so without an explicit release every slow
embed would sit on an idle-in-transaction connection and a slow Ollama would
exhaust the pool. app/api/v1/search.py therefore commits the read-only auth
transaction before calling the embedder.

The embedder is replaced by one that records the session's state at the moment
it is called; that is the only point at which the property matters.
"""
from app.api.v1 import search as search_module
from tests.factories import make_api_key, make_note, make_passage
from tests.fakes import fake_embedding


def test_no_transaction_is_open_while_the_query_is_embedded(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    passage = make_passage(
        db_session, note, ordinal=0, content="The pelicanium isotope decays slowly."
    )
    seen = []

    def recording_embedder(query):
        # Recorded, not asserted here: an assertion error would be swallowed
        # into the route's error handling and surface as an unrelated 500.
        seen.append(db_session.in_transaction())
        return fake_embedding(query)

    monkeypatch.setattr(search_module, "get_query_embedding", recording_embedder)

    response = client.post(
        "/api/v1/search",
        json={"q": "pelicanium", "top_k": 5},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200, response.text
    assert seen == [False]
    # The repository re-begins its own transaction afterwards, so its set_config
    # read-back guard still passes and the seeded rows are still visible.
    body = response.json()
    assert body["degraded"] is False
    assert [r["passage_id"] for r in body["results"]][0] == passage.id
