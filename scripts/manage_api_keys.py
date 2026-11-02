# scripts/manage_api_keys.py
"""Create, list and revoke API keys without SQL.

    uv run python scripts/manage_api_keys.py create <client_name>
    uv run python scripts/manage_api_keys.py list [--client NAME] [--include-revoked]
    uv run python scripts/manage_api_keys.py revoke --id N
    uv run python scripts/manage_api_keys.py revoke --client NAME --yes

The secret is shown once, by `create`. `list` shows the id, client, display
prefix and timestamps, never the secret or its hash. Revoking sets revoked_at
rather than deleting the row, which keeps the audit trail. Revoking a key that
is already revoked reports 0 and exits 0.

Exit codes: 0 success (including "revoked 0"), 1 refused (--client without
--yes) or no API key with that --id. Argument errors are argparse's (2).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.db import get_sessionmaker
from app.core.security import generate_api_key
from app.models.api_key import ApiKey

EXIT_OK = 0
EXIT_REFUSED = 1

SessionFactory = Callable[[], AbstractContextManager[Session]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="mint a key for a client (shown once)")
    create.add_argument("client_name")

    listing = commands.add_parser("list", help="list keys, never their secrets")
    listing.add_argument("--client", dest="client_name", default=None)
    listing.add_argument("--include-revoked", action="store_true")

    revoke = commands.add_parser("revoke", help="set revoked_at on one key or on a client's keys")
    selector = revoke.add_mutually_exclusive_group(required=True)
    selector.add_argument("--id", dest="key_id", type=int)
    selector.add_argument("--client", dest="client_name")
    revoke.add_argument(
        "--yes", action="store_true", help="required with --client: it revokes every active key of that client"
    )
    return parser


def _cmd_create(db: Session, client_name: str) -> int:
    plaintext, key_hash, prefix = generate_api_key(client_name)
    db.add(ApiKey(key_prefix=prefix, key_hash=key_hash, client_name=client_name))
    db.commit()

    print(f"client_name : {client_name}")
    print(f"api_key     : {plaintext}")
    print("⚠️  Store this now — it cannot be retrieved again.")
    return EXIT_OK


def _timestamp(value) -> str:
    return "-" if value is None else value.isoformat(timespec="seconds")


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows, strict=True)]
    template = "  ".join(f"{{:<{width}}}" for width in widths)
    print(template.format(*headers).rstrip())
    for row in rows:
        print(template.format(*row).rstrip())


def _cmd_list(db: Session, client_name: str | None, include_revoked: bool) -> int:
    # Named columns, not the ORM entity: key_hash is never selected.
    stmt = select(
        ApiKey.id,
        ApiKey.client_name,
        ApiKey.key_prefix,
        ApiKey.created_at,
        ApiKey.revoked_at,
    ).order_by(ApiKey.id)
    if client_name is not None:
        stmt = stmt.where(ApiKey.client_name == client_name)
    if not include_revoked:
        stmt = stmt.where(ApiKey.revoked_at.is_(None))

    rows = db.execute(stmt).all()
    if not rows:
        print("no API keys")
        return EXIT_OK

    _print_table(
        ["ID", "CLIENT", "PREFIX", "CREATED", "REVOKED"],
        [
            [
                str(row.id),
                row.client_name,
                row.key_prefix,
                _timestamp(row.created_at),
                _timestamp(row.revoked_at),
            ]
            for row in rows
        ],
    )
    return EXIT_OK


def _cmd_revoke(db: Session, key_id: int | None, client_name: str | None, yes: bool) -> int:
    if key_id is not None:
        exists = db.execute(select(ApiKey.id).where(ApiKey.id == key_id)).scalar_one_or_none()
        if exists is None:
            print(f"no API key with id {key_id}", file=sys.stderr)
            return EXIT_REFUSED
        target = ApiKey.id == key_id
    else:
        if not yes:
            print(
                f"refusing to revoke every active key of client {client_name!r} without --yes",
                file=sys.stderr,
            )
            return EXIT_REFUSED
        target = ApiKey.client_name == client_name

    # revoked_at IS NULL makes a repeat revoke touch nothing, so it reports 0.
    result = db.execute(
        update(ApiKey)
        .where(ApiKey.revoked_at.is_(None), target)
        .values(revoked_at=func.now())
    )
    db.commit()
    print(f"revoked {result.rowcount}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None, session_factory: SessionFactory | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    factory = session_factory or get_sessionmaker()

    with factory() as db:
        if args.command == "create":
            return _cmd_create(db, args.client_name)
        if args.command == "list":
            return _cmd_list(db, args.client_name, args.include_revoked)
        return _cmd_revoke(db, args.key_id, args.client_name, args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
