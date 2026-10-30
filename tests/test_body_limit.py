"""app.core.body_limit.BodySizeLimitMiddleware: the /api/v2 request-byte bound.

The contract under test (see the module docstring in app/core/body_limit.py):

- only paths under ``/api/v2/`` are inspected at all;
- an over-limit ``Content-Length`` is rejected with 413 and the exact detail
  string, *without* the wrapped app ever being entered -- which is the whole
  point, because FastAPI parses a JSON body before any dependency, including
  auth, runs;
- a malformed or negative ``Content-Length`` is a 400;
- a body sent without ``Content-Length`` (chunked) is counted as it arrives and
  rejected the moment the running total passes the limit;
- anything within the limit passes through with its bytes intact.

Most tests here drive the middleware over a tiny recording app with a small
``max_bytes``, so nothing has to send 8 MiB. The constructor takes ``max_bytes``
precisely so this is possible. Two tests at the bottom go through the real
application to prove the middleware is actually wired into create_app() and
sits ahead of authentication.
"""
import asyncio
import json

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from app.core.body_limit import MAX_V2_BODY_BYTES, BodySizeLimitMiddleware
from tests.factories import make_api_key

LIMIT = 100

TOO_LARGE_DETAIL = f"Request body too large: limit is {LIMIT} bytes."
BAD_LENGTH_DETAIL = "Invalid Content-Length header."


class Recorder:
    """What the wrapped app saw, if it was entered at all."""

    def __init__(self) -> None:
        self.handler_calls = 0
        self.dependency_calls = 0
        self.body: bytes | None = None
        self.saw_content_length: bool | None = None

    @property
    def entered(self) -> bool:
        return self.handler_calls > 0 or self.dependency_calls > 0


def _build_app(max_bytes: int = LIMIT) -> tuple[TestClient, Recorder]:
    """A minimal app behind the middleware, plus a record of what it saw.

    The dependency is recorded separately from the handler: a rejection must
    happen before *either* runs, and in the real app the dependency slot is
    where authentication lives.
    """
    recorder = Recorder()
    app = FastAPI()

    def record_dependency() -> None:
        recorder.dependency_calls += 1

    async def echo(request: Request) -> dict[str, int]:
        recorder.handler_calls += 1
        recorder.body = await request.body()
        recorder.saw_content_length = "content-length" in request.headers
        return {"received": len(recorder.body)}

    app.add_api_route(
        "/api/v2/echo",
        echo,
        methods=["POST"],
        dependencies=[Depends(record_dependency)],
    )
    app.add_api_route(
        "/api/v1/echo",
        echo,
        methods=["POST"],
        dependencies=[Depends(record_dependency)],
    )

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_bytes)
    return TestClient(app), recorder


def _raw_asgi_post(path: str, headers: list[tuple[bytes, bytes]], body: bytes):
    """Send one request straight at the middleware over the ASGI interface.

    TestClient/httpx own the ``Content-Length`` header for most requests, so a
    deliberately malformed or negative value is set here instead, where nothing
    can normalise it away. Returns (status, parsed_json_body, recorder).
    """
    recorder = Recorder()

    async def inner_app(scope, receive, send) -> None:
        recorder.handler_calls += 1
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"{}"})

    middleware = BodySizeLimitMiddleware(inner_app, max_bytes=LIMIT)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }

    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message) -> None:
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return status, json.loads(payload), recorder


def _chunks(total_bytes: int, chunk_size: int = 32):
    """A generator body, so httpx uses chunked encoding and no Content-Length."""

    def generate():
        remaining = total_bytes
        while remaining > 0:
            size = min(chunk_size, remaining)
            yield b"x" * size
            remaining -= size

    return generate()


# --- Content-Length path -----------------------------------------------------


def test_an_over_limit_content_length_is_rejected_with_413():
    client, recorder = _build_app()

    response = client.post("/api/v2/echo", content=b"x" * (LIMIT + 1))

    assert response.status_code == 413
    assert response.json() == {"detail": TOO_LARGE_DETAIL}


def test_an_over_limit_content_length_never_reaches_the_app():
    """The rejection must precede the wrapped app, hence any dependency."""
    client, recorder = _build_app()

    client.post("/api/v2/echo", content=b"x" * (LIMIT + 1))

    assert not recorder.entered, (
        "the wrapped app was entered for an oversized body; in the real app "
        "that would mean auth (and JSON parsing) ran first"
    )


def test_the_detail_reports_the_configured_limit_not_the_default():
    client, _ = _build_app(max_bytes=64)

    response = client.post("/api/v2/echo", content=b"x" * 65)

    assert response.json() == {
        "detail": "Request body too large: limit is 64 bytes."
    }


def test_a_body_exactly_at_the_limit_passes_through_intact():
    client, recorder = _build_app()
    body = b"y" * LIMIT

    response = client.post("/api/v2/echo", content=body)

    assert response.status_code == 200
    assert response.json() == {"received": LIMIT}
    assert recorder.body == body


def test_a_body_under_the_limit_passes_through_intact():
    client, recorder = _build_app()
    body = b"under the limit" * 3

    response = client.post("/api/v2/echo", content=body)

    assert response.status_code == 200
    assert recorder.body == body
    assert recorder.handler_calls == 1
    assert recorder.dependency_calls == 1


def test_an_empty_body_passes_through():
    client, recorder = _build_app()

    response = client.post("/api/v2/echo", content=b"")

    assert response.status_code == 200
    assert recorder.body == b""


def test_a_malformed_content_length_is_rejected_with_400():
    status, payload, recorder = _raw_asgi_post(
        "/api/v2/echo",
        [(b"content-length", b"not-a-number")],
        b"",
    )

    assert status == 400
    assert payload == {"detail": BAD_LENGTH_DETAIL}
    assert not recorder.entered


def test_a_negative_content_length_is_rejected_with_400():
    status, payload, recorder = _raw_asgi_post(
        "/api/v2/echo",
        [(b"content-length", b"-1")],
        b"",
    )

    assert status == 400
    assert payload == {"detail": BAD_LENGTH_DETAIL}
    assert not recorder.entered


def test_a_forged_under_limit_content_length_does_not_buy_a_bigger_body():
    """A client that lies about its length is still bounded by real bytes."""
    status, payload, recorder = _raw_asgi_post(
        "/api/v2/echo",
        [(b"content-length", b"10")],
        b"z" * (LIMIT + 1),
    )

    assert status == 413
    assert payload == {"detail": TOO_LARGE_DETAIL}
    assert not recorder.entered


# --- chunked / streamed path -------------------------------------------------


def test_a_streamed_body_really_arrives_without_a_content_length():
    """Guards the premise of the chunked tests below."""
    client, recorder = _build_app()

    response = client.post("/api/v2/echo", content=_chunks(LIMIT))

    assert response.status_code == 200
    assert recorder.saw_content_length is False, (
        "httpx sent a Content-Length for a generator body, so the tests below "
        "would no longer exercise the byte-counting path"
    )


def test_an_over_limit_chunked_body_is_rejected_with_413():
    client, recorder = _build_app()

    response = client.post("/api/v2/echo", content=_chunks(LIMIT * 4))

    assert response.status_code == 413
    assert response.json() == {"detail": TOO_LARGE_DETAIL}
    assert not recorder.entered


def test_a_chunked_body_under_the_limit_passes_through_intact():
    client, recorder = _build_app()

    response = client.post("/api/v2/echo", content=_chunks(LIMIT - 4))

    assert response.status_code == 200
    assert response.json() == {"received": LIMIT - 4}
    assert recorder.body == b"x" * (LIMIT - 4)


# --- scoping -----------------------------------------------------------------


def test_a_non_v2_path_is_not_intercepted():
    client, recorder = _build_app()
    body = b"x" * (LIMIT * 5)

    response = client.post("/api/v1/echo", content=body)

    assert response.status_code == 200
    assert recorder.body == body


def test_a_non_v2_path_with_a_bad_content_length_is_not_intercepted():
    status, _, recorder = _raw_asgi_post(
        "/api/v1/echo",
        [(b"content-length", b"not-a-number")],
        b"",
    )

    assert status == 200
    assert recorder.handler_calls == 1


# --- the limit's own derivation ----------------------------------------------


def test_the_limit_admits_the_largest_body_the_content_cap_allows():
    """Pins the derivation written in app/core/body_limit.py's comment.

    MAX_CONTENT_CHARS is 500,000 characters, and the worst case per character
    is an astral-plane codepoint, which json.dumps escapes as a surrogate pair
    -- two \\uXXXX escapes, 12 bytes. The byte limit must not reject a body the
    character check would have accepted.
    """
    from app.api.v2.ingest import MAX_CONTENT_CHARS

    worst_case = json.dumps({"content": "\U0001F600" * MAX_CONTENT_CHARS})

    assert len(worst_case.encode("utf-8")) <= MAX_V2_BODY_BYTES


# --- through the real application --------------------------------------------


OVERSIZE = MAX_V2_BODY_BYTES + 1


def test_an_unauthenticated_oversized_ingest_gets_413_not_401(client):
    """The middleware must beat auth, which it can only do by running first.

    The Content-Length is forged over a tiny real body rather than really
    sending 8 MiB: httpx lets an explicitly supplied header win over its own
    computed one, and the middleware rejects on the declared value alone,
    before reading the body at all. No X-API-Key is sent.
    """
    response = client.post(
        "/api/v2/ingest",
        content=b"{}",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(OVERSIZE),
        },
    )

    assert response.status_code == 413, (
        f"expected 413 before auth, got {response.status_code}: "
        f"{response.text}"
    )
    assert response.json() == {
        "detail": f"Request body too large: limit is {MAX_V2_BODY_BYTES} bytes."
    }


def test_a_normal_authenticated_ingest_still_succeeds(client, db_session):
    """The middleware must be transparent to an ordinary request."""
    key = make_api_key(db_session, "acme")

    response = client.post(
        "/api/v2/ingest",
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
    """A non-/api/v2 route keeps working with the middleware installed."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
