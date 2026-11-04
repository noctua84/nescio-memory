"""A stdio MCP server over the memory service's HTTP API.

Its only way in is the API, using one client's own key, so it can never see or
touch more than that client could with curl. It imports nothing from `app`:
that package's settings require DATABASE_URL and belong to the API process.

Importing this package must keep working without the optional `mcp` extra. Only
nescio_mcp.server imports the SDK, and __main__ imports that module inside a
try, so that `python -m nescio_mcp` can say what is missing instead of raising.
"""
