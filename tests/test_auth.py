"""Authentication is enforced on the api_router itself (app/api/v2/router.py),
so these assertions hold for every /api/v2 route. They are run against each
protected route rather than one representative, so a route that is ever
mounted outside the authenticated router fails here by name.
"""
import pytest

from tests.factories import make_api_key

SEARCH_PAYLOAD = {"q": "anything", "top_k": 5}
INGEST_PAYLOAD = {
    "scope": "repo_a",
    "path_or_uri": "docs/auth-probe.md",
    "content": "# Heron ledger\n\nThe heron ledger tracks quarterly widget counts.",
}

# (method, path, json body). The notes routes take no body; a note id that
# cannot exist is fine because auth must reject before the lookup runs.
PROTECTED_ROUTES = [
    pytest.param("POST", "/api/v2/search", SEARCH_PAYLOAD, id="search"),
    pytest.param("POST", "/api/v2/ingest", INGEST_PAYLOAD, id="ingest"),
    pytest.param("GET", "/api/v2/notes/999999", None, id="get-note"),
    pytest.param("DELETE", "/api/v2/notes/999999", None, id="delete-note"),
]


def _send(client, method, path, payload, headers=None):
    return client.request(method, path, json=payload, headers=headers or {})


@pytest.mark.parametrize(("method", "path", "payload"), PROTECTED_ROUTES)
def test_request_without_an_api_key_is_rejected(client, method, path, payload):
    response = _send(client, method, path, payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API key"


@pytest.mark.parametrize(("method", "path", "payload"), PROTECTED_ROUTES)
def test_unknown_api_key_is_rejected(client, method, path, payload):
    response = _send(
        client,
        method,
        path,
        payload,
        headers={"X-API-Key": "nm_definitely_not_a_real_key"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or revoked API key"


@pytest.mark.parametrize(("method", "path", "payload"), PROTECTED_ROUTES)
def test_revoked_api_key_is_rejected(client, db_session, method, path, payload):
    key = make_api_key(db_session, "acme", revoked=True)
    response = _send(client, method, path, payload, headers={"X-API-Key": key})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or revoked API key"


def test_valid_api_key_reaches_the_search_endpoint(client, db_session):
    key = make_api_key(db_session, "acme")
    response = client.post(
        "/api/v2/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )
    assert response.status_code == 200
    # An exact match rather than a subset check: the point of this test is
    # that an authenticated search reaches the endpoint and returns its whole
    # empty-corpus body. degraded is False because the conftest `client`
    # fixture fakes the query embedding, so the vector arm runs.
    assert response.json() == {"results": [], "degraded": False}


def test_valid_api_key_reaches_the_ingest_endpoint(client, db_session):
    key = make_api_key(db_session, "acme")
    response = client.post(
        "/api/v2/ingest", json=INGEST_PAYLOAD, headers={"X-API-Key": key}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"note_id", "passages", "skipped"}
    assert isinstance(body["note_id"], int)
    assert body["passages"] >= 1
    assert body["skipped"] is False


def test_valid_api_key_reaches_the_notes_endpoint(client, db_session):
    # A 404 with the route's own detail proves auth passed and the handler
    # ran; a rejected request would have been a 401 instead.
    key = make_api_key(db_session, "acme")
    response = client.get("/api/v2/notes/999999", headers={"X-API-Key": key})
    assert response.status_code == 404
    assert response.json() == {"detail": "Note not found"}


@pytest.mark.parametrize(("method", "path", "payload"), PROTECTED_ROUTES)
def test_only_the_plaintext_key_authenticates_not_the_stored_hash(
    client, db_session, method, path, payload
):
    # Guards against a regression where the stored hash is compared directly to
    # the header, which would make the database contents themselves usable as
    # credentials.
    from app.core.security import _hash_key

    key = make_api_key(db_session, "acme")
    response = _send(
        client, method, path, payload, headers={"X-API-Key": _hash_key(key)}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or revoked API key"
