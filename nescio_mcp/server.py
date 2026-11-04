"""The MCP tools, each one call on the HTTP API.

Tools are plain functions, which the SDK runs on a worker thread, so a blocking
request or a retry wait does not stall the protocol loop.

stdout is the protocol channel here. Nothing in this package may write to it,
and logging goes to stderr, which is where the SDK's configure_logging puts it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from nescio_mcp.api_client import NescioApiClient, NescioApiError
from nescio_mcp.config import McpSettings

logger = logging.getLogger(__name__)

_READ_ONLY = ToolAnnotations(read_only_hint=True)
_DESTRUCTIVE = ToolAnnotations(destructive_hint=True)


def build_server(settings: McpSettings) -> MCPServer:
    api = NescioApiClient(
        settings.build_http_client(),
        max_retry_after_s=settings.max_retry_after_s,
    )
    return create_server(api)


def create_server(api: NescioApiClient) -> MCPServer:
    """The tools over a given API client, so the tests can inject theirs."""
    # WARNING, not the SDK's INFO: at INFO httpx logs every request URL, and a
    # delete by path has the note's path in its query string. A failed tool call
    # still reaches the caller as a tool error.
    server = MCPServer(
        "nescio-memory",
        instructions=(
            "Long-term memory: search, store, fetch and delete notes. Every call "
            "acts as one client and sees only that client's notes."
        ),
        log_level="WARNING",
    )

    @server.tool(annotations=_READ_ONLY)
    def search_memory(
        query: str, top_k: int = 5, repo_filter: str | None = None
    ) -> dict[str, Any]:
        """Search stored notes by keyword and by meaning.

        Returns passages ordered by relevance, up to top_k (1..50). Trailing
        results can be weak matches, as there is no relevance floor. repo_filter
        limits the search to one scope.

        When `degraded` is true the query could not be embedded and the results
        are keyword matches only, so an empty or poor result does not mean the
        memory holds nothing relevant.
        """
        return _call(
            "search_memory", lambda: api.search(query, top_k=top_k, repo_filter=repo_filter)
        )

    @server.tool()
    def ingest_note(
        scope: str, path_or_uri: str, content: str, force: bool = False
    ) -> dict[str, Any]:
        """Store a Markdown note, replacing any note at the same scope and path.

        scope groups notes and may be empty, which is the root scope. path_or_uri
        must be a relative path without '..', or an absolute URI. Unchanged
        content is skipped (`skipped` is true) unless force is set.
        """
        return _call(
            "ingest_note",
            lambda: api.ingest(scope, path_or_uri, content, force=force),
        )

    @server.tool(annotations=_READ_ONLY)
    def get_note(id: int) -> dict[str, Any]:
        """Fetch one note by id, with its full body and passage count."""
        return _call("get_note", lambda: api.get_note(id))

    @server.tool(annotations=_DESTRUCTIVE)
    def delete_note(id: int) -> dict[str, Any]:
        """Permanently delete one note and its passages by id."""

        def delete() -> dict[str, Any]:
            api.delete_note(id)
            return {"deleted": True, "id": id}

        return _call("delete_note", delete)

    @server.tool(annotations=_DESTRUCTIVE)
    def delete_note_by_path(scope: str, path_or_uri: str) -> dict[str, Any]:
        """Permanently delete the note at a scope and path, and its passages.

        scope may be empty, which is the root scope. Returns what was removed.
        """
        return _call(
            "delete_note_by_path", lambda: api.delete_note_by_path(scope, path_or_uri)
        )

    @server.tool(annotations=_READ_ONLY)
    def memory_status() -> dict[str, Any]:
        """Report whether the memory service is alive and able to serve.

        `health` is liveness. `ready.status` is `ready`, `degraded` (the embedding
        backend is down, so searches are keyword-only and ingests fail) or
        `not_ready` (the database is down).
        """
        return _call(
            "memory_status", lambda: {"health": api.health(), "ready": api.ready()}
        )

    return server


def _call[T](tool: str, operation: Callable[[], T]) -> T:
    """Run an API call, so that the only thing a tool can raise is a ToolError.

    The SDK withholds the text of any other exception from the client, but it
    logs that exception in full, and an exception's text is not ours to vouch
    for. So those are reduced to a type name here, for the log and the caller.
    """
    try:
        return operation()
    except NescioApiError as exc:
        raise ToolError(str(exc)) from None
    except Exception as exc:  # noqa: BLE001 -- the point: nothing else may escape
        logger.error("tool %s failed unexpectedly: %s", tool, type(exc).__name__)
        raise ToolError(
            f"The MCP server failed unexpectedly ({type(exc).__name__}); see its stderr log."
        ) from None
