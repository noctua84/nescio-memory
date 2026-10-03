"""Authentication is enforced on the api_router itself, so these assertions
hold for every /api/v1 route, not just the one used here.
"""
from tests.factories import make_api_key

SEARCH_PAYLOAD = {"query": "anything", "top_k": 5}


def test_request_without_an_api_key_is_rejected(client):
    response = client.post("/api/v1/search", json=SEARCH_PAYLOAD)
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API key"


def test_unknown_api_key_is_rejected(client):
    response = client.post(
        "/api/v1/search",
        json=SEARCH_PAYLOAD,
        headers={"X-API-Key": "nm_definitely_not_a_real_key"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or revoked API key"


def test_revoked_api_key_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme", revoked=True)
    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or revoked API key"


def test_valid_api_key_reaches_the_endpoint(client, db_session):
    key = make_api_key(db_session, "acme")
    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )
    assert response.status_code == 200
    # Still an exact match rather than a subset check: the point of this test
    # is that an authenticated search reaches the endpoint and returns its
    # whole empty-corpus body, and `contexts` is part of that body now.
    # `contexts` is empty because the default `context` mode is "none".
    assert response.json() == {"results": [], "contexts": []}


def test_only_the_plaintext_key_authenticates_not_the_stored_hash(client, db_session):
    # Guards against a regression where the stored hash is compared directly to
    # the header, which would make the database contents themselves usable as
    # credentials.
    from app.core.security import _hash_key

    key = make_api_key(db_session, "acme")
    response = client.post(
        "/api/v1/search",
        json=SEARCH_PAYLOAD,
        headers={"X-API-Key": _hash_key(key)},
    )
    assert response.status_code == 401
