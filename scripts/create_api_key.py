# scripts/create_api_key.py
"""Mint an API key. A thin shim over `manage_api_keys.py create`.

    uv run python scripts/create_api_key.py my-client

Kept so the documented command keeps working. Omit the argument and the client
is named `default`, as it always was.
"""
import sys
from collections.abc import Sequence

from scripts.manage_api_keys import SessionFactory
from scripts.manage_api_keys import main as manage_api_keys_main


def main(argv: Sequence[str] | None = None, session_factory: SessionFactory | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    client_name = args[0] if args else "default"
    return manage_api_keys_main(["create", client_name], session_factory=session_factory)


if __name__ == "__main__":
    raise SystemExit(main())
