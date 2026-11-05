"""Run the MCP server over stdio: uv run --extra mcp python -m nescio_mcp

Run it from the repository root: it imports scripts._cli_safety from there.

Exit code 2 is a usage error, as in scripts/reingest.py: the extra is not
installed or the configuration is refused. A client launching this sees the
process die, and only stderr says why, so both cases have to say it plainly.
"""
from __future__ import annotations

import sys

from pydantic import ValidationError

from nescio_mcp.config import McpSettings, configuration_errors

EXIT_USAGE_ERROR = 2


def main() -> int:
    try:
        from nescio_mcp.server import build_server
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] != "mcp":
            raise
        print(
            "nescio_mcp: the MCP SDK is not installed. "
            "Run it with: uv run --extra mcp python -m nescio_mcp",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    try:
        settings = McpSettings()
    except ValidationError as exc:
        for line in configuration_errors(exc):
            print(f"nescio_mcp: {line}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    build_server(settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
