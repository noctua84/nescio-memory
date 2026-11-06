"""A thin typed wrapper over the memory service's HTTP API.

It takes an already-built client and validates nothing about where that client
points: McpSettings checks the URL at configuration time, and the tests inject
a TestClient on http://testserver. The key travels as a default header of that
client, so nothing here ever handles it except to keep it out of a message.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

import httpx

API_KEY_HEADER = "X-API-Key"

DEFAULT_MAX_RETRY_AFTER_S = 30.0

# 429 is rate limiting and 503 a dependency that may recover. Every other status
# fails the same way on a second try.
_RETRYABLE_STATUSES = frozenset({429, 503})

# The wait when a retryable response has no usable Retry-After, which is what
# scripts/reingest.py waits too.
_DEFAULT_RETRY_DELAY_S = 5.0

# A detail reaches a model's context, and a 422 lists every failed field. The
# server's own messages are far shorter; this bounds what a proxy's error page
# or a very long validation list can cost.
_MAX_ERROR_DETAIL_CHARS = 1000
_MAX_RETRY_AFTER_ECHO_CHARS = 64


class NescioApiError(Exception):
    """A request that failed. str() is safe to show: it never holds the key."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class NescioApiClient:
    """One method per API operation. Each returns the decoded JSON body.

    `http` is annotated as httpx.Client and used only through its public
    request API. With the `mcp` extra installed, starlette's TestClient is built
    on httpx2 rather than httpx, and it works here unchanged.

    A 429 or 503 is retried once, after min(Retry-After, max_retry_after_s),
    the rule scripts/reingest.py uses. Unattended callers get a known worst case
    per call: two request timeouts plus one capped wait.
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        max_retry_after_s: float = DEFAULT_MAX_RETRY_AFTER_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._max_retry_after_s = max_retry_after_s
        self._sleep = sleep

    def search(
        self, query: str, top_k: int = 5, repo_filter: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"q": query, "top_k": top_k}
        if repo_filter is not None:
            payload["repo_filter"] = repo_filter
        return _json_object(self._request("POST", "/api/v1/search", json=payload))

    def ingest(
        self, scope: str, path_or_uri: str, content: str, force: bool = False
    ) -> dict[str, Any]:
        payload = {
            "scope": scope,
            "path_or_uri": path_or_uri,
            "content": content,
            "force": force,
        }
        return _json_object(self._request("POST", "/api/v1/ingest", json=payload))

    def get_note(self, note_id: int) -> dict[str, Any]:
        return _json_object(self._request("GET", f"/api/v1/notes/{note_id}"))

    def delete_note(self, note_id: int) -> None:
        # 204: there is no body to decode.
        self._request("DELETE", f"/api/v1/notes/{note_id}")

    def delete_note_by_path(self, scope: str, path_or_uri: str) -> dict[str, Any]:
        params = {"scope": scope, "path_or_uri": path_or_uri}
        return _json_object(self._request("DELETE", "/api/v1/notes", params=params))

    def health(self) -> dict[str, Any]:
        response = self._send("GET", "/health")
        if not response.is_success:
            raise self._error(response)
        return _json_object(response)

    def ready(self) -> dict[str, Any]:
        """The readiness report, whether or not the service is ready.

        A 503 carrying the report is the answer to the question, so it is
        returned rather than raised, and never retried: waiting up to the cap to
        ask again would only make a status check slower. A 503 without one is
        some other failure, such as a proxy's error page.
        """
        response = self._send("GET", "/ready")
        if response.status_code == 503:
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and "status" in body:
                return body
        if not response.is_success:
            raise self._error(response)
        return _json_object(response)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = self._send(method, path, **kwargs)
        if response.status_code in _RETRYABLE_STATUSES:
            self._sleep(self._retry_delay(response.headers.get("Retry-After")))
            response = self._send(method, path, **kwargs)
        if not response.is_success:
            raise self._error(response)
        return response

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            # The type only, and no chaining: an exception's text is not ours to
            # vouch for. h11 puts an illegal header value in its message, and
            # the only header value that varies here is the key.
            raise NescioApiError(
                f"Could not complete the request to the memory service ({type(exc).__name__})."
            ) from None

    def _retry_delay(self, retry_after: str | None) -> float:
        delay = _DEFAULT_RETRY_DELAY_S
        if retry_after is not None:
            try:
                parsed = float(retry_after)
            except ValueError:
                parsed = math.nan
            if parsed >= 0 and math.isfinite(parsed):
                delay = parsed
        return min(delay, self._max_retry_after_s)

    def _error(self, response: httpx.Response) -> NescioApiError:
        detail = self._scrub(_detail_of(response))[:_MAX_ERROR_DETAIL_CHARS]
        message = f"HTTP {response.status_code}: {detail}"
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            retry_after = self._scrub(retry_after)[:_MAX_RETRY_AFTER_ECHO_CHARS]
            message += f" (Retry-After: {retry_after})"
        return NescioApiError(
            message, status_code=response.status_code, retry_after=retry_after
        )

    def _scrub(self, text: str) -> str:
        """Take the key out of text a server sent back.

        The service never echoes it, but whatever answers at that address might:
        a proxy's debug page, say. The text goes to a model, so this is not left
        to the other side's good behaviour.
        """
        key = self._http.headers.get(API_KEY_HEADER)
        return text.replace(key, "[redacted]") if key else text


def _detail_of(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        # A string everywhere except 422, which is a list of error objects.
        return detail if isinstance(detail, str) else json.dumps(detail)
    return json.dumps(body)


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise NescioApiError(
            f"HTTP {response.status_code}: the memory service did not return a JSON object."
        )
    return body
