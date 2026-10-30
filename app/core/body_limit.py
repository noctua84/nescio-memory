"""Bounds the raw byte size of /api/v2 request bodies before anything parses them.

FastAPI reads and parses a JSON body before any dependency (including auth)
runs, so an unauthenticated client could otherwise send a multi-GB body and
have it buffered in full by the worker. v1 does not need this: it takes
``Form(...)`` fields, and Starlette already enforces a 1 MiB per-form-field
limit ahead of any application code.

This is a pure ASGI middleware (not ``BaseHTTPMiddleware``) specifically so it
can inspect and reject a request directly off the ``receive`` channel, before
ever calling into the wrapped app (and therefore before any dependency,
including auth, runs) -- something BaseHTTPMiddleware's request-wrapping
cannot do without first assembling the body into a single bytes object.
"""
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# 500,000 chars of content (MAX_CONTENT_CHARS in app/api/v2/ingest.py) can cost
# up to 12 bytes per char once JSON-encoded: a `\uXXXX` escape is 6 bytes, and
# a character outside the Basic Multilingual Plane is encoded as a surrogate
# pair -- two such escapes, i.e. 12 bytes -- per source character. That caps
# the `content` field alone at 6 MB. The remaining ~2 MB of headroom covers
# `meta` and the rest of the JSON envelope (field names, quoting, structure),
# while staying well short of a limit that would ever reject a request the
# 500k-char check in ingest.py would otherwise accept.
MAX_V2_BODY_BYTES = 8 * 1024 * 1024

_V2_PREFIX = "/api/v2/"

_BAD_CONTENT_LENGTH_BODY = b'{"detail": "Invalid Content-Length header."}'


def _too_large_body(limit: int) -> bytes:
    return (
        b'{"detail": "Request body too large: limit is '
        + str(limit).encode("ascii")
        + b' bytes."}'
    )


async def _send_json(send: Send, status_code: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class BodySizeLimitMiddleware:
    """Pure ASGI middleware that rejects oversized /api/v2 request bodies.

    When ``Content-Length`` is present, an over-limit value is rejected
    immediately, before ``self.app`` -- and therefore any dependency inside
    it, including auth -- is ever invoked.

    Without a usable ``Content-Length`` (e.g. chunked transfer encoding),
    bytes are counted as ``receive()`` delivers them. The middleware stops
    pulling further chunks the moment the running total passes the limit and
    responds with 413 directly, again without calling ``self.app``.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_V2_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(_V2_PREFIX):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")

        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await _send_json(send, 400, _BAD_CONTENT_LENGTH_BODY)
                return
            if declared < 0:
                await _send_json(send, 400, _BAD_CONTENT_LENGTH_BODY)
                return
            if declared > self.max_bytes:
                await _send_json(send, 413, _too_large_body(self.max_bytes))
                return
            # Content-Length is within bounds. A client could still lie and
            # send more than it declared, so the byte-counting buffer below
            # runs regardless of whether Content-Length was present.

        # Buffer messages ourselves, counting bytes as they arrive, so the
        # running total can be checked before anything is handed to the app.
        # This is what lets an oversized chunked body be rejected without
        # ever letting a downstream dependency (e.g. auth) run, and without
        # pulling any further chunks off `receive()` once the limit is
        # passed.
        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await _send_json(send, 413, _too_large_body(self.max_bytes))
                return
            if not message.get("more_body", False):
                break

        index = 0

        async def replay_receive() -> Message:
            nonlocal index
            if index < len(buffered):
                message = buffered[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)
