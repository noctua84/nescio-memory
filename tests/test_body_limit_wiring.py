"""BodySizeLimitMiddleware's byte limit and its wiring into the real app.

Covers the two things the recording-app tests in `test_body_limit.py` cannot:
that the limit is large enough for the biggest body the ingest content cap
allows (the derivation written in app/core/body_limit.py), and that the
middleware is actually installed in create_app() ahead of authentication, so an
unauthenticated oversized ingest is a 413 rather than a 401 and an ordinary
authenticated ingest and /health are untouched.
"""

import json

from app.core.body_limit import MAX_BODY_BYTES
from tests.factories import make_api_key

# --- the limit's own derivation ----------------------------------------------


def test_the_limit_admits_the_largest_body_the_content_cap_allows():
    """Pins the derivation written in app/core/body_limit.py's comment.

    MAX_CONTENT_CHARS is 500,000 characters, and the worst case per character
    is an astral-plane codepoint, which json.dumps escapes as a surrogate pair
    -- two \\uXXXX escapes, 12 bytes. The byte limit must not reject a body the
    character check would have accepted.
    """
    from app.api.v1.ingest import MAX_CONTENT_CHARS

    worst_case = json.dumps({"content": "\U0001f600" * MAX_CONTENT_CHARS})

    assert len(worst_case.encode("utf-8")) <= MAX_BODY_BYTES


# --- through the real application --------------------------------------------


OVERSIZE = MAX_BODY_BYTES + 1


def test_an_unauthenticated_oversized_ingest_gets_413_not_401(client):
    """The middleware must beat auth, which it can only do by running first.

    The Content-Length is forged over a tiny real body rather than really
    sending 8 MiB: httpx lets an explicitly supplied header win over its own
    computed one, and the middleware rejects on the declared value alone,
    before reading the body at all. No X-API-Key is sent.
    """
    response = client.post(
        "/api/v1/ingest",
        content=b"{}",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(OVERSIZE),
        },
    )

    assert response.status_code == 413, (
        f"expected 413 before auth, got {response.status_code}: {response.text}"
    )
    assert response.json() == {
        "detail": f"Request body too large: limit is {MAX_BODY_BYTES} bytes."
    }


def test_a_normal_authenticated_ingest_still_succeeds(client, db_session):
    """The middleware must be transparent to an ordinary request."""
    key = make_api_key(db_session, "acme")

    response = client.post(
        "/api/v1/ingest",
        json={
            "scope": "repo_a",
            "path_or_uri": "docs/note.md",
            "content": "The marmoset throughput ledger for the quarter.",
        },
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200, response.text
    assert response.json()["passages"] >= 1
    assert response.json()["skipped"] is False


def test_health_is_not_intercepted(client):
    """A non-/api/v1 route keeps working with the middleware installed."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
