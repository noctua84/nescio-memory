"""The MCP tools, through the SDK's own client and against the real app.

Every tool call goes through `mcp.Client`, so what is asserted is what a model
would see: the tool's result or its is_error text. The end-to-end tests inject a
TestClient into NescioApiClient and run against the real routes and Postgres.

429 and the retry cases use httpx.MockTransport instead, as the real app has no
way to produce a 429 followed by a 200 on demand.

Skipped without the `mcp` extra, and the skip shows in every run's summary
(addopts = "-ra"). CI installs the extra, so these do run there.
"""
import asyncio
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import httpx
from fastapi.testclient import TestClient
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult

from app.api.v1 import search as search_module
from app.api.v1.ingest import MAX_CONTENT_CHARS
from app.config import settings
from app.core import readiness
from app.core.errors import EmbeddingBackendError
from nescio_mcp.api_client import NescioApiClient
from nescio_mcp.config import McpSettings
from nescio_mcp.server import build_server, create_server
from tests.factories import make_api_key

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY = "nm_test_s3cret_mcp_key"
TOOLS = {
    "search_memory",
    "ingest_note",
    "get_note",
    "delete_note",
    "delete_note_by_path",
    "memory_status",
}
NOTE = "The pelicanium isotope decays slowly in cool storage."


def _call(server, tool: str, **arguments) -> CallToolResult:
    async def run() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(run())


def _text(result: CallToolResult) -> str:
    return " ".join(block.text for block in result.content)


def _ok(server, tool: str, **arguments) -> dict:
    result = _call(server, tool, **arguments)
    assert not result.is_error, _text(result)
    return result.structured_content


def _error(server, tool: str, **arguments) -> str:
    result = _call(server, tool, **arguments)
    assert result.is_error, f"expected a tool error, got {result.structured_content}"
    return _text(result)


@pytest.fixture
def server_for(client):
    """A server holding one key, on a client for the test's app and transaction.

    A TestClient per key, since the key is a default header of the client, as it
    is in production. They all share the app, and so the `client` fixture's
    database override and fake embeddings.
    """

    def build(key: str):
        http = TestClient(client.app, headers={"X-API-Key": key})
        return create_server(NescioApiClient(http))

    return build


@pytest.fixture
def server(server_for, db_session):
    return server_for(make_api_key(db_session, "acme"))


# --- every tool, end to end ------------------------------------------------------


def test_ingest_search_get_delete_by_path_then_a_404(server):
    ingested = _ok(
        server, "ingest_note", scope="repo_a", path_or_uri="docs/iso.md", content=NOTE
    )
    assert ingested["skipped"] is False
    assert ingested["passages"] >= 1

    found = _ok(server, "search_memory", query="pelicanium", top_k=3)
    assert found["degraded"] is False
    assert [hit["path_or_uri"] for hit in found["results"]] == ["docs/iso.md"]
    assert found["results"][0]["note_id"] == ingested["note_id"]

    note = _ok(server, "get_note", id=ingested["note_id"])
    assert note["body"] == NOTE
    assert note["scope"] == "repo_a"

    deleted = _ok(
        server, "delete_note_by_path", scope="repo_a", path_or_uri="docs/iso.md"
    )
    assert deleted["notes_deleted"] == 1
    assert deleted["notes"][0]["note_id"] == ingested["note_id"]

    assert "HTTP 404: Note not found" in _error(server, "get_note", id=ingested["note_id"])
    assert "HTTP 404: Note not found" in _error(
        server, "delete_note_by_path", scope="repo_a", path_or_uri="docs/iso.md"
    )


def test_delete_note_by_id(server):
    note_id = _ok(
        server, "ingest_note", scope="", path_or_uri="docs/iso.md", content=NOTE
    )["note_id"]

    assert _ok(server, "delete_note", id=note_id) == {"deleted": True, "id": note_id}
    assert "HTTP 404" in _error(server, "delete_note", id=note_id)


def test_force_reingests_unchanged_content(server):
    arguments = {"scope": "repo_a", "path_or_uri": "docs/iso.md", "content": NOTE}
    _ok(server, "ingest_note", **arguments)

    assert _ok(server, "ingest_note", **arguments)["skipped"] is True
    assert _ok(server, "ingest_note", **arguments, force=True)["skipped"] is False


def test_repo_filter_limits_the_search_to_one_scope(server):
    _ok(server, "ingest_note", scope="repo_a", path_or_uri="a.md", content=NOTE)
    _ok(server, "ingest_note", scope="repo_b", path_or_uri="b.md", content=NOTE)

    found = _ok(server, "search_memory", query="pelicanium", repo_filter="repo_b")

    assert [hit["path_or_uri"] for hit in found["results"]] == ["b.md"]


def test_top_k_reaches_the_server(server):
    for path in ("a.md", "b.md"):
        _ok(server, "ingest_note", scope="repo_a", path_or_uri=path, content=NOTE)

    assert len(_ok(server, "search_memory", query="pelicanium", top_k=1)["results"]) == 1
    assert "HTTP 422" in _error(server, "search_memory", query="pelicanium", top_k=51)


def test_a_degraded_search_says_so(server, monkeypatch):
    _ok(server, "ingest_note", scope="repo_a", path_or_uri="a.md", content=NOTE)

    def refuse(_query):
        raise EmbeddingBackendError("refused")

    monkeypatch.setattr(search_module, "get_query_embedding", refuse)

    found = _ok(server, "search_memory", query="pelicanium")

    assert found["degraded"] is True
    assert [hit["path_or_uri"] for hit in found["results"]] == ["a.md"]


def test_the_server_limits_come_back_as_their_detail(server):
    # Large content is the server's to refuse; the tool passes its 400 on.
    message = _error(
        server,
        "ingest_note",
        scope="repo_a",
        path_or_uri="big.md",
        content="x" * (MAX_CONTENT_CHARS + 1),
    )

    assert "HTTP 400: content is too large" in message


def test_a_bad_path_comes_back_as_its_detail(server):
    message = _error(
        server, "ingest_note", scope="repo_a", path_or_uri="../up.md", content=NOTE
    )

    assert "HTTP 400: path_or_uri must be a relative path" in message


def test_memory_status_reports_health_and_readiness(server, engine, monkeypatch):
    monkeypatch.setattr(readiness, "get_engine", lambda: engine)
    # The local backend is in-process, so nothing remote is contacted.
    monkeypatch.setattr(settings, "embedding_backend", "local")
    monkeypatch.setattr(settings, "ready_cache_ttl_s", 0.0)

    status = _ok(server, "memory_status")

    assert status["health"]["status"] == "ok"
    assert status["ready"]["status"] == "ready"
    assert status["ready"]["checks"]["database"]["status"] == "ok"


# --- tenant isolation -------------------------------------------------------------


def test_a_tool_holding_one_key_cannot_touch_another_clients_notes(server_for, db_session):
    alice = server_for(make_api_key(db_session, "alice"))
    bob = server_for(make_api_key(db_session, "bob"))
    note_id = _ok(
        alice, "ingest_note", scope="repo_a", path_or_uri="docs/iso.md", content=NOTE
    )["note_id"]

    assert _ok(bob, "search_memory", query="pelicanium")["results"] == []
    assert "HTTP 404" in _error(bob, "get_note", id=note_id)
    assert "HTTP 404" in _error(bob, "delete_note", id=note_id)
    assert "HTTP 404" in _error(
        bob, "delete_note_by_path", scope="repo_a", path_or_uri="docs/iso.md"
    )

    # Same path and scope under another tenant is a different note.
    bobs = _ok(bob, "ingest_note", scope="repo_a", path_or_uri="docs/iso.md", content="other")
    assert bobs["note_id"] != note_id

    # Nothing bob did reached alice's note.
    assert _ok(alice, "get_note", id=note_id)["body"] == NOTE
    assert len(_ok(alice, "search_memory", query="pelicanium")["results"]) == 1


def test_an_unknown_key_gets_the_server_detail_and_nothing_else(server_for):
    message = _error(server_for(KEY), "search_memory", query="pelicanium")

    assert "HTTP 401: Invalid or revoked API key" in message
    assert KEY not in message


# --- 429 and 503 -------------------------------------------------------------------


def _stubbed(
    handler: Callable[[httpx.Request], httpx.Response],
    max_retry_after_s: float = 30.0,
):
    """A server on a stub transport. Returns it, its sleeps and its requests."""
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    http = httpx.Client(
        transport=httpx.MockTransport(recording),
        base_url="http://testserver",
        headers={"X-API-Key": KEY},
    )
    api = NescioApiClient(http, max_retry_after_s=max_retry_after_s, sleep=sleeps.append)
    return create_server(api), sleeps, requests


def _sequence(*responses: httpx.Response):
    remaining = list(responses)
    return lambda request: remaining.pop(0)


def _rejected(status: int, retry_after: str | None = None, detail: str = "slow down"):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return httpx.Response(status, json={"detail": detail}, headers=headers)


SEARCH_OK = {"results": [], "degraded": False}


@pytest.mark.parametrize("status", [429, 503])
def test_a_retryable_status_is_retried_once_after_retry_after(status):
    server, sleeps, requests = _stubbed(
        _sequence(_rejected(status, "7"), httpx.Response(200, json=SEARCH_OK))
    )

    assert _ok(server, "search_memory", query="q") == SEARCH_OK
    assert sleeps == [7.0]
    assert len(requests) == 2


@pytest.mark.parametrize("status", [429, 503])
def test_a_second_retryable_status_is_a_tool_error_with_retry_after(status):
    server, sleeps, requests = _stubbed(
        _sequence(_rejected(status, "7"), _rejected(status, "9", "still busy"))
    )

    message = _error(server, "search_memory", query="q")

    assert f"HTTP {status}: still busy" in message
    assert "Retry-After: 9" in message
    # Once, and no more.
    assert sleeps == [7.0]
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("retry_after", "cap", "expected"),
    [
        ("3600", 30.0, 30.0),
        ("3600", 0.0, 0.0),
        ("2", 30.0, 2.0),
        ("0", 30.0, 0.0),
        # Unusable, so the same 5 seconds scripts/reingest.py waits.
        (None, 30.0, 5.0),
        ("soon", 30.0, 5.0),
        ("-3", 30.0, 5.0),
        ("nan", 30.0, 5.0),
        ("inf", 30.0, 5.0),
        ("Wed, 21 Oct 2026 07:28:00 GMT", 30.0, 5.0),
        # Which the cap still bounds.
        (None, 1.0, 1.0),
    ],
)
def test_the_wait_is_capped(retry_after, cap, expected):
    server, sleeps, _ = _stubbed(
        _sequence(_rejected(429, retry_after), httpx.Response(200, json=SEARCH_OK)),
        max_retry_after_s=cap,
    )

    _ok(server, "search_memory", query="q")

    assert sleeps == [expected]


@pytest.mark.parametrize("status", [400, 401, 404, 413, 500, 502])
def test_any_other_status_is_not_retried(status):
    server, sleeps, requests = _stubbed(_sequence(_rejected(status, detail="nope")))

    assert f"HTTP {status}: nope" in _error(server, "search_memory", query="q")
    assert sleeps == []
    assert len(requests) == 1


def test_a_retry_after_on_a_non_retryable_status_is_still_reported():
    server, sleeps, _ = _stubbed(_sequence(_rejected(500, "12", "broken")))

    assert "Retry-After: 12" in _error(server, "get_note", id=1)
    assert sleeps == []


def test_a_redirect_is_an_error_and_is_not_followed():
    server, _, requests = _stubbed(
        lambda request: httpx.Response(307, headers={"Location": "http://evil.example/"})
    )

    assert "HTTP 307" in _error(server, "get_note", id=1)
    assert len(requests) == 1


def test_a_list_detail_is_shown_as_json():
    body = {"detail": [{"loc": ["body", "q"], "msg": "required", "type": "missing"}]}
    server, _, _ = _stubbed(lambda request: httpx.Response(422, json=body))

    message = _error(server, "search_memory", query="q")

    assert "HTTP 422" in message
    assert '"msg": "required"' in message


def test_a_huge_error_body_is_truncated():
    server, _, _ = _stubbed(lambda request: httpx.Response(502, text="x" * 100_000))

    assert len(_error(server, "get_note", id=1)) < 1200


def test_a_successful_response_that_is_not_a_json_object_is_a_tool_error():
    server, _, _ = _stubbed(lambda request: httpx.Response(200, text="<html>"))

    assert "did not return a JSON object" in _error(server, "get_note", id=1)


def test_the_status_tool_treats_a_not_ready_503_as_its_answer():
    report = {"status": "not_ready", "checks": {"database": {"status": "fail"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503, json=report)

    server, sleeps, requests = _stubbed(handler)

    assert _ok(server, "memory_status") == {"health": {"status": "ok"}, "ready": report}
    assert sleeps == []
    assert len(requests) == 2


def test_the_status_tool_reports_a_503_that_is_not_a_readiness_report():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503, text="bad gateway")

    server, _, _ = _stubbed(handler)

    assert "HTTP 503: bad gateway" in _error(server, "memory_status")


# --- the key never appears in error text --------------------------------------------


def _raises(exc: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


@pytest.mark.parametrize(
    "handler",
    [
        # A server or proxy echoing the header back, in each place one could.
        lambda request: httpx.Response(
            400, json={"detail": f"bad key {request.headers['X-API-Key']}"}
        ),
        lambda request: httpx.Response(
            400, json={"detail": [{"input": request.headers["X-API-Key"]}]}
        ),
        lambda request: httpx.Response(502, text=f"echo: {request.headers['X-API-Key']}"),
        lambda request: httpx.Response(
            400, json={"detail": "x"}, headers={"Retry-After": request.headers["X-API-Key"]}
        ),
        # Exceptions quoting it, as h11 does with an illegal header value.
        _raises(httpx.LocalProtocolError(f"Illegal header value b'{KEY}'")),
        _raises(httpx.ConnectTimeout(f"timed out sending {KEY}")),
        _raises(RuntimeError(f"unexpected {KEY}")),
    ],
    ids=[
        "string-detail",
        "list-detail",
        "plain-text-body",
        "retry-after-header",
        "h11-style-exception",
        "transport-exception",
        "non-httpx-exception",
    ],
)
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search_memory", {"query": "q"}),
        ("ingest_note", {"scope": "s", "path_or_uri": "p.md", "content": "c"}),
        ("get_note", {"id": 1}),
        ("delete_note", {"id": 1}),
        ("delete_note_by_path", {"scope": "s", "path_or_uri": "p.md"}),
        ("memory_status", {}),
    ],
)
def test_the_key_is_not_in_the_error_text_or_the_log(handler, tool, arguments, caplog):
    server, _, _ = _stubbed(handler)

    with caplog.at_level("DEBUG"):
        message = _error(server, tool, **arguments)

    assert KEY not in message
    assert KEY not in caplog.text


def test_a_transport_failure_is_a_tool_error_naming_the_failure():
    server, sleeps, requests = _stubbed(_raises(httpx.ConnectError("refused")))

    assert "ConnectError" in _error(server, "search_memory", query="q")
    assert sleeps == []
    assert len(requests) == 1


# --- the protocol ---------------------------------------------------------------------


def _listed_tools(server) -> dict:
    async def run():
        async with Client(server) as client:
            return await client.list_tools()

    return {tool.name: tool for tool in asyncio.run(run()).tools}


def _offline_server():
    return _stubbed(lambda request: httpx.Response(500))[0]


def test_the_tools_are_exactly_the_ones_decided():
    # An exact set: scope delete and bulk sync are deliberately not tools.
    assert set(_listed_tools(_offline_server())) == TOOLS


def test_the_deletes_are_destructive_and_the_reads_are_read_only():
    tools = _listed_tools(_offline_server())

    for name in ("delete_note", "delete_note_by_path"):
        assert tools[name].annotations.destructive_hint is True
    for name in ("search_memory", "get_note", "memory_status"):
        assert tools[name].annotations.read_only_hint is True
    assert tools["ingest_note"].annotations is None


def test_every_tool_has_a_description_and_the_decided_parameters():
    tools = _listed_tools(_offline_server())

    assert all(tool.description for tool in tools.values())
    expected = {
        "search_memory": ({"query"}, {"top_k", "repo_filter"}),
        "ingest_note": ({"scope", "path_or_uri", "content"}, {"force"}),
        "get_note": ({"id"}, set()),
        "delete_note": ({"id"}, set()),
        "delete_note_by_path": ({"scope", "path_or_uri"}, set()),
        "memory_status": (set(), set()),
    }
    for name, (required, optional) in expected.items():
        schema = tools[name].input_schema
        assert set(schema.get("required", [])) == required, name
        assert set(schema["properties"]) == required | optional, name


def test_build_server_uses_the_settings(monkeypatch):
    monkeypatch.setenv("NESCIO_URL", "http://localhost:1")
    monkeypatch.setenv("NESCIO_API_KEY", KEY)

    server = build_server(McpSettings())

    assert set(_listed_tools(server)) == TOOLS


# --- the real process, over real stdio ------------------------------------------------


def _process_env(**extra: str) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name != "DATABASE_URL" and not name.upper().startswith("NESCIO_")
    }
    return env | extra


def test_the_process_speaks_the_protocol_on_stdout():
    # Anything else written to stdout corrupts the stream and this fails. Port 1
    # refuses, so the tool call also proves a dead service is a tool error and
    # not a crash. No DATABASE_URL: importing app.config would stop the process.
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nescio_mcp"],
        env=_process_env(NESCIO_URL="http://127.0.0.1:1", NESCIO_API_KEY=KEY),
        cwd=REPO_ROOT,
    )

    async def run():
        async with Client(parameters) as client:
            listed = await client.list_tools()
            return listed, await client.call_tool("search_memory", {"query": "q"})

    listed, result = asyncio.run(run())

    assert {tool.name for tool in listed.tools} == TOOLS
    assert result.is_error
    assert "ConnectError" in _text(result)
    assert KEY not in _text(result)


def test_a_refused_configuration_stops_the_process_and_says_why_on_stderr():
    result = subprocess.run(
        [sys.executable, "-m", "nescio_mcp"],
        cwd=REPO_ROOT,
        env=_process_env(NESCIO_URL="http://evil.example", NESCIO_API_KEY=KEY),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "NESCIO_URL" in result.stderr
    assert KEY not in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_configuration_names_every_variable():
    result = subprocess.run(
        [sys.executable, "-m", "nescio_mcp"],
        cwd=REPO_ROOT,
        env=_process_env(),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "NESCIO_URL" in result.stderr
    assert "NESCIO_API_KEY" in result.stderr
