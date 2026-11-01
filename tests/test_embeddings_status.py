"""Ollama HTTP status classification: an allow-list of transient statuses.

Only 408, 429, 500, 502, 503 and 504 mean "try again later" and stay on the
exact base class, EmbeddingBackendError (search degrades, ingest answers 503 +
Retry-After). Every other non-2xx -- redirects included, since httpx does not
follow them and raise_for_status() raises for 3xx -- is a fault retrying never
fixes, and raises EmbeddingBackendMisconfigured (500, no Retry-After, never
degraded). Statuses observed against Ollama: 404 for a model never pulled, 501
for a chat model, 307 for a trailing-slash OLLAMA_URL, 308 behind an
http->https proxy.

The unit tests need no database: httpx.Client is replaced by a stub. The route
tests at the bottom use the `client` fixture with the real embedding chain
restored, so the classification is exercised end to end.
"""
import re

import httpx
import pytest
from sqlalchemy import func, select

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.embeddings import get_embeddings, get_query_embedding
from app.core.errors import (
    RETRY_AFTER_SECONDS,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key, make_note, make_passage

# Caller data that must never surface in an exception message: these messages
# are logged, and the texts are note content or search queries.
SECRET = "c7e2d0-caller-supplied-text-that-must-not-be-echoed"

# Synthetic, and distinct from every default, so finding it in a message can
# only mean the message was built from settings.ollama_url.
OLLAMA_URL = "http://embedder.invalid:11434/api/embed"

TRANSIENT_STATUSES = [408, 429, 500, 502, 503, 504]
MISCONFIGURED_STATUSES = [
    301, 302, 307, 308,
    400, 401, 403, 404, 405, 413, 422,
    501, 505,
]

MISCONFIGURED_DETAIL = "Embedding backend is misconfigured"
UNAVAILABLE_DETAIL = "Embedding backend unavailable"


class _StubClient:
    """Stands in for httpx.Client; every post() answers with `behaviour()`."""

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
    """An Ollama-like answer with `code`; 3xx carry a Location, as Ollama's do."""
    headers = {"location": OLLAMA_URL} if 300 <= code < 400 else {}
    return lambda: httpx.Response(
        status_code=code,
        json={"error": "synthetic"},
        headers=headers,
        request=httpx.Request("POST", OLLAMA_URL + "/"),
    )


def _install(monkeypatch, code) -> _StubClient:
    stub = _StubClient(_status(code))
    monkeypatch.setattr(embeddings_module.httpx, "Client", stub)
    return stub


@pytest.fixture(autouse=True)
def _ollama_backend(monkeypatch):
    # Pinned so a developer's EMBEDDING_BACKEND=local or OLLAMA_URL cannot
    # change what these tests exercise.
    monkeypatch.setattr(settings, "embedding_backend", "ollama")
    monkeypatch.setattr(settings, "ollama_url", OLLAMA_URL)


# --- unit: the classification itself ------------------------------------------


@pytest.mark.parametrize("status", TRANSIENT_STATUSES)
@pytest.mark.parametrize(
    "embed",
    [lambda text: get_embeddings([text]), get_query_embedding],
    ids=["ingest-path", "query-path"],
)
def test_a_transient_status_stays_on_the_exact_base_class(
    monkeypatch, status, embed
):
    stub = _install(monkeypatch, status)

    with pytest.raises(EmbeddingBackendError) as excinfo:
        embed(SECRET)

    assert stub.posts == 1, "the stubbed backend was never reached"
    # Exact type: pytest.raises matches subclasses, so this is what catches a
    # 429 or 504 wrongly reported as a misconfiguration.
    assert type(excinfo.value) is EmbeddingBackendError
    message = str(excinfo.value)
    assert re.search(rf"\bHTTP {status}\b", message)
    assert SECRET not in message


@pytest.mark.parametrize("status", MISCONFIGURED_STATUSES)
@pytest.mark.parametrize(
    "embed",
    [lambda text: get_embeddings([text]), get_query_embedding],
    ids=["ingest-path", "query-path"],
)
def test_every_other_status_is_a_misconfiguration(monkeypatch, status, embed):
    stub = _install(monkeypatch, status)

    with pytest.raises(EmbeddingBackendMisconfigured) as excinfo:
        embed(SECRET)

    assert stub.posts == 1, "the stubbed backend was never reached"
    assert type(excinfo.value) is EmbeddingBackendMisconfigured
    message = str(excinfo.value)
    # Word-bounded so the port 11434 cannot satisfy it by accident.
    assert re.search(rf"\b{status}\b", message)
    # The URL is what an operator fixes for a redirect, so it must be named.
    assert OLLAMA_URL in message
    assert SECRET not in message


def test_the_allow_list_is_closed_across_every_3xx_4xx_and_5xx(monkeypatch):
    # The lists above are samples; this is the allow-list property itself. A
    # deny-list that forgot a status (a 506 from a proxy, say) would leave it
    # transient and make search degrade for good.
    wrong = {}
    for status in range(300, 600):
        _install(monkeypatch, status)
        try:
            get_embeddings([SECRET])
        except EmbeddingBackendError as exc:
            expected = (
                EmbeddingBackendError
                if status in TRANSIENT_STATUSES
                else EmbeddingBackendMisconfigured
            )
            if type(exc) is not expected:
                wrong[status] = type(exc).__name__
        else:
            wrong[status] = "no exception"

    assert wrong == {}


# --- routes: the classification reaches the HTTP answer -----------------------


def _use_real_chain(monkeypatch):
    """Undo the `client` fixture's fakes so the real embedding chain runs."""
    monkeypatch.setattr(
        search_module, "get_query_embedding", embeddings_module.get_query_embedding
    )
    monkeypatch.setattr(
        ingest_module, "get_embeddings", embeddings_module.get_embeddings
    )


def _ingest(client, key):
    return client.post(
        "/api/v1/ingest",
        json={
            "scope": "repo_a",
            "path_or_uri": "docs/note.md",
            "content": "The pelicanium isotope decays slowly in cool storage.",
        },
        headers={"X-API-Key": key},
    )


def _search(client, key):
    return client.post(
        "/api/v1/search",
        json={"q": "pelicanium", "top_k": 10},
        headers={"X-API-Key": key},
    )


def _assert_nothing_persisted(db_session):
    # The route never committed, so a real request teardown discards its
    # writes; rolling the session back reproduces that.
    db_session.rollback()
    assert db_session.scalar(select(func.count()).select_from(Note)) == 0
    assert db_session.scalar(select(func.count()).select_from(Passage)) == 0


@pytest.mark.parametrize("status", TRANSIENT_STATUSES)
def test_a_transient_status_degrades_search_to_full_text(
    client, db_session, monkeypatch, status
):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    keyword = make_passage(
        db_session, note, ordinal=0, content="The pelicanium isotope decays."
    )
    make_passage(db_session, note, ordinal=1, content="Marmosets like warmth.")
    _use_real_chain(monkeypatch)
    stub = _install(monkeypatch, status)

    response = _search(client, key)

    assert stub.posts == 1, "the stubbed backend was never reached"
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] is True
    assert [r["passage_id"] for r in body["results"]] == [keyword.id]


@pytest.mark.parametrize("status", [404, 501, 307, 308])
def test_a_permanent_status_on_ingest_is_500_without_retry_after(
    client, db_session, monkeypatch, status
):
    key = make_api_key(db_session, "acme")
    _use_real_chain(monkeypatch)
    stub = _install(monkeypatch, status)

    response = _ingest(client, key)

    assert stub.posts >= 1, "the stubbed backend was never reached"
    assert response.status_code == 500
    assert response.json() == {"detail": MISCONFIGURED_DETAIL}
    assert "Retry-After" not in response.headers
    _assert_nothing_persisted(db_session)


@pytest.mark.parametrize("status", [429, 503, 504])
def test_a_transient_status_on_ingest_is_503_with_retry_after(
    client, db_session, monkeypatch, status
):
    key = make_api_key(db_session, "acme")
    _use_real_chain(monkeypatch)
    stub = _install(monkeypatch, status)

    response = _ingest(client, key)

    assert stub.posts >= 1, "the stubbed backend was never reached"
    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE_DETAIL}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
    _assert_nothing_persisted(db_session)
