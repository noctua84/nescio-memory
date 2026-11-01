"""POST /api/v1/search: when it degrades to full-text, and when it must not.

Only the exact transient class, EmbeddingBackendError, degrades: the backend is
briefly unreachable and keyword results are still a useful answer. Its
subclasses mean the backend answered wrongly (BadResponse, 503 + Retry-After) or
cannot work as deployed (Misconfigured, 500, no Retry-After), and serving
FTS-only there would hide a fault that needs attention. Ingest never degrades.

Two search tests leave the real get_query_embedding -> _embed -> _fetch_batch ->
_validated chain in place and stub only httpx.Client or _fetch_batch (the
permanent-status test is parametrized over four codes), so the classification is
exercised end to end rather than restated. No test here reaches the network:
every path that could is stubbed.
"""
import httpx
import pytest
from sqlalchemy import event, func, select

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.errors import (
    RETRY_AFTER_SECONDS,
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key, make_note, make_passage
from tests.fakes import fake_embedding

BAD_RESPONSE_DETAIL = "Embedding backend returned an unexpected response"
MISCONFIGURED_DETAIL = "Embedding backend is misconfigured"
UNAVAILABLE_DETAIL = "Embedding backend unavailable"


def _search(client, key, q="pelicanium", top_k=10):
    return client.post(
        "/api/v1/search",
        json={"q": q, "top_k": top_k},
        headers={"X-API-Key": key},
    )


def _seed(db):
    """One passage FTS finds for 'pelicanium', and one it cannot find.

    With the vector arm on, search fills up to top_k, so both come back; with
    it off, only the keyword match can. That difference is what makes a
    degraded body observably FTS-only rather than merely labelled so.
    """
    note = make_note(db, "acme")
    keyword = make_passage(
        db, note, ordinal=0, content="The pelicanium isotope decays slowly."
    )
    other = make_passage(
        db, note, ordinal=1, content="Marmosets prefer warm enclosures."
    )
    return keyword, other


def _raising(exception):
    calls = []

    def embedder(query):
        calls.append(query)
        raise exception

    embedder.calls = calls
    return embedder


class _StubClient:
    """Stands in for httpx.Client, as in tests/test_embeddings.py."""

    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.posts = 0

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        self.posts += 1
        return self._behaviour()


def _status(code):
    return lambda: httpx.Response(
        status_code=code,
        json={"error": "synthetic"},
        request=httpx.Request("POST", "http://ollama.invalid/api/embed"),
    )


def _use_real_query_chain(monkeypatch):
    """Undo the `client` fixture's fake so the real embedding chain runs."""
    monkeypatch.setattr(settings, "embedding_backend", "ollama")
    monkeypatch.setattr(
        search_module, "get_query_embedding", embeddings_module.get_query_embedding
    )


# --- degraded: the exact transient class only --------------------------------


def test_a_transient_failure_returns_200_degraded_with_fts_only_results(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    keyword, _ = _seed(db_session)
    embedder = _raising(EmbeddingBackendError("backend down"))
    monkeypatch.setattr(search_module, "get_query_embedding", embedder)

    response = _search(client, key)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] is True
    assert len(embedder.calls) == 1
    # The keyword match only: the non-matching passage would be here if any
    # vector arm had run, because search fills up to top_k.
    assert [r["passage_id"] for r in body["results"]] == [keyword.id]
    assert all(r["vector_rank"] is None for r in body["results"])
    assert body["results"][0]["fts_rank"] == 1


def test_a_bad_response_is_503_with_retry_after_and_not_degraded(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    _seed(db_session)
    monkeypatch.setattr(
        search_module,
        "get_query_embedding",
        _raising(EmbeddingBackendBadResponse("garbled")),
    )

    response = _search(client, key)

    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


def test_a_misconfiguration_is_500_without_retry_after_and_not_degraded(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    _seed(db_session)
    monkeypatch.setattr(
        search_module,
        "get_query_embedding",
        _raising(EmbeddingBackendMisconfigured("wrong model")),
    )

    response = _search(client, key)

    assert response.status_code == 500
    assert response.json() == {"detail": MISCONFIGURED_DETAIL}
    assert "Retry-After" not in response.headers


@pytest.mark.parametrize("code", [404, 501, 307, 308])
def test_a_permanent_status_from_ollama_on_the_real_chain_is_500_not_degraded(
    client, db_session, monkeypatch, code
):
    # What Ollama was observed to answer: 404 for a model that was never
    # pulled, 501 for a chat (non-embedding) model, 307 for an OLLAMA_URL with
    # a trailing slash; 308 is the http->https proxy redirect. httpx follows
    # neither redirect. None is ever fixed by retrying, so degrading here would
    # serve keyword results for good while claiming to be transient. The full
    # status allow-list is in tests/test_embeddings_status.py.
    key = make_api_key(db_session, "acme")
    _seed(db_session)
    _use_real_query_chain(monkeypatch)
    stub = _StubClient(_status(code))
    monkeypatch.setattr(embeddings_module.httpx, "Client", stub)

    response = _search(client, key)

    assert stub.posts == 1, "the stubbed backend was never reached"
    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": MISCONFIGURED_DETAIL}
    assert "degraded" not in body
    assert "Retry-After" not in response.headers


def test_a_zero_norm_query_vector_on_the_real_chain_is_503_not_200(
    client, db_session, monkeypatch
):
    # A zero vector is the right width and type, so only _validated's norm
    # check stands between it and a cosine distance of NaN.
    key = make_api_key(db_session, "acme")
    _seed(db_session)
    _use_real_query_chain(monkeypatch)
    batches = []

    def zero_batch(texts, timeout):
        batches.append(list(texts))
        return [[0.0] * settings.embedding_dimension for _ in texts]

    monkeypatch.setattr(embeddings_module, "_fetch_batch", zero_batch)

    response = _search(client, key)

    assert batches, "the real chain never reached _fetch_batch"
    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


# --- not degraded -------------------------------------------------------------


def test_the_happy_path_is_not_degraded_and_runs_the_vector_arm(
    client, db_session
):
    key = make_api_key(db_session, "acme")
    keyword, other = _seed(db_session)

    response = _search(client, key)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] is False
    # Both passages: the vector arm ran and filled up to top_k.
    assert {r["passage_id"] for r in body["results"]} == {keyword.id, other.id}
    assert all(r["vector_rank"] is not None for r in body["results"])


def test_vector_weight_zero_never_calls_the_embedder_and_is_not_degraded(
    client, db_session, monkeypatch, fts_only
):
    key = make_api_key(db_session, "acme")
    keyword, _ = _seed(db_session)
    # It would raise if called, so a call would show up as a degraded body.
    embedder = _raising(EmbeddingBackendError("must not be called"))
    monkeypatch.setattr(search_module, "get_query_embedding", embedder)

    response = _search(client, key)

    assert response.status_code == 200, response.text
    body = response.json()
    assert embedder.calls == []
    assert body["degraded"] is False
    assert [r["passage_id"] for r in body["results"]] == [keyword.id]


# --- when the embed happens ---------------------------------------------------
# The degraded warning's content is tested in tests/test_search_degraded_log.py.


def test_the_query_is_embedded_before_any_search_guard_statement(
    client, db_session, connection, monkeypatch
):
    # The embed must not overlap the search's statement_timeout window, which
    # the repository opens with its first statement, the set_config guard.
    key = make_api_key(db_session, "acme")
    _seed(db_session)
    events: list[tuple[str, str]] = []

    def recording_embedder(query):
        events.append(("embed", query))
        return fake_embedding(query)

    def record_statement(conn, cursor, statement, parameters, context, many):
        events.append(("sql", statement))

    monkeypatch.setattr(search_module, "get_query_embedding", recording_embedder)
    event.listen(connection, "before_cursor_execute", record_statement)
    try:
        response = _search(client, key)
    finally:
        event.remove(connection, "before_cursor_execute", record_statement)

    assert response.status_code == 200, response.text
    embed_at = [i for i, (kind, _) in enumerate(events) if kind == "embed"]
    guard_at = [
        i
        for i, (kind, statement) in enumerate(events)
        if kind == "sql" and "set_config" in statement
    ]
    assert len(embed_at) == 1
    assert guard_at, "no set_config statement was recorded; the probe saw nothing"
    assert embed_at[0] < min(guard_at)


# --- ingest does not degrade --------------------------------------------------


def test_ingest_with_a_transient_embedding_failure_is_503_and_persists_nothing(
    client, db_session, monkeypatch
):
    # Search degrades on this exact class; ingest must not, since a note stored
    # without vectors would be silently unsearchable by meaning. Real chain,
    # with the connection refused as when Ollama is down.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(settings, "embedding_backend", "ollama")
    monkeypatch.setattr(
        ingest_module, "get_embeddings", embeddings_module.get_embeddings
    )

    def refuse():
        raise httpx.ConnectError(
            "refused", request=httpx.Request("POST", "http://ollama.invalid/")
        )

    stub = _StubClient(refuse)
    monkeypatch.setattr(embeddings_module.httpx, "Client", stub)

    response = client.post(
        "/api/v1/ingest",
        json={
            "scope": "repo_a",
            "path_or_uri": "docs/note.md",
            "content": "The pelicanium isotope decays slowly in cool storage.",
        },
        headers={"X-API-Key": key},
    )

    assert stub.posts >= 1, "the stubbed backend was never reached"
    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE_DETAIL}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
    # The route never committed, so a real request teardown discards its
    # writes; rolling the session back reproduces that. Had the route
    # committed, the rows would survive this rollback.
    db_session.rollback()
    assert db_session.scalar(select(func.count()).select_from(Note)) == 0
    assert db_session.scalar(select(func.count()).select_from(Passage)) == 0
