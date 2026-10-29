"""Re-ingest a corpus of Markdown notes into the memory service.

    uv run python -m scripts.reingest --root <dir> --api-key <key>
        [--url http://localhost:8080] [--scope <name>] [--exclude GLOB]...
        [--no-default-excludes] [--force] [--dry-run] [--timeout 120]

Invoked with -m, not as a path: `python scripts/reingest.py` puts scripts/ on
sys.path instead of the repository root, so `import app` fails.

Exit codes: 0 nothing failed, 1 one or more files failed to ingest, 2 a
usage or IO error (for example an unreadable --root).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import httpx

# These two defaults are not a convenience -- they are a retrieval-quality
# requirement. MEMORY.md is a generated index of other notes' descriptions
# and learning-log.md is an 82 KB promotion ledger; both beat original notes
# on exact-keyword FTS, so leaving them in the corpus poisons search results.
# Exclusion is therefore additive: --exclude only ever adds more globs, and
# re-admitting these two requires the explicit, visible --no-default-excludes
# flag rather than someone silently replacing the default list with their own.
DEFAULT_EXCLUDES = ["**/MEMORY.md", "**/learning-log.md"]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE_ERROR = 2


def resolve_excludes(user_excludes: Sequence[str], use_defaults: bool) -> list[str]:
    """Combine the default exclusion globs with any user-supplied ones.

    Additive by design: with use_defaults=True (the normal case) the result
    always contains DEFAULT_EXCLUDES plus whatever the caller added, never a
    replacement of them.
    """
    if not use_defaults:
        return list(user_excludes)
    return [*DEFAULT_EXCLUDES, *user_excludes]


def _match(rel: str, pattern: str) -> bool:
    """Match rel against pattern the way a "**/NAME" exclude glob should.

    pathlib.PurePath.match() treats "**" as a wildcard for exactly one path
    component, not pathlib's own "zero or more directories" glob semantics
    (that richer behaviour only exists on PurePath.full_match(), added in
    Python 3.13; this project supports 3.12+). Left as-is, "**/learning-
    log.md" would therefore fail to match a top-level learning-log.md,
    silently re-admitting the exact file the default excludes exist to
    keep out. Stripping a leading "**/" and retrying covers the "zero
    directories" case while still using PurePath.match() for everything
    else.
    """
    if Path(rel).match(pattern):
        return True
    if pattern.startswith("**/"):
        return _match(rel, pattern[3:])
    return False


def select_files(root: Path, excludes: Sequence[str]) -> list[Path]:
    """Return every *.md file under root not matching any exclude glob.

    Matching uses the posix-style relative path from root, so a glob
    written as "**/MEMORY.md" matches regardless of the host's path
    separator. Side-effect free: only reads the directory tree, never the
    file contents.
    """
    files = sorted(root.rglob("*.md"))
    selected = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if any(_match(rel, pattern) for pattern in excludes):
            continue
        selected.append(path)
    return selected


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reingest",
        description=(
            "Re-ingest a directory tree of Markdown notes via POST "
            "/api/v2/ingest. Exclusion is additive: the defaults "
            f"{DEFAULT_EXCLUDES!r} are always applied unless "
            "--no-default-excludes is passed, and every --exclude adds to "
            "them rather than replacing them."
        ),
    )
    parser.add_argument("--root", required=True, type=Path, help="Directory to walk recursively for *.md files.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key sent as the X-API-Key header. Falls back to the NESCIO_API_KEY environment variable.",
    )
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the memory service.")
    parser.add_argument("--scope", default="", help="scope value sent with every ingest request.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Additional glob (matched with PurePath.match against the posix "
            "relative path from --root) to exclude. Adds to the defaults "
            f"{DEFAULT_EXCLUDES!r}; repeatable."
        ),
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help=(
            "Drop the default exclusions "
            f"{DEFAULT_EXCLUDES!r} instead of adding to them. These two files "
            "poison keyword search results, so this is the explicit, visible "
            "way to re-admit them."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Pass force=true to every ingest request.")
    parser.add_argument("--dry-run", action="store_true", help="List the files that would be posted and exit without any HTTP call.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds (default: 120).")
    return parser.parse_args(argv)


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key
    return os.environ.get("NESCIO_API_KEY")


def _ingest_one(
    client: httpx.Client,
    url: str,
    api_key: str,
    scope: str,
    rel_path: str,
    content: str,
    force: bool,
) -> tuple[str, int]:
    """POST one file. Returns (outcome, passages) where outcome is one of
    "ok", "skipped" or "FAILED <status> <detail>".
    """
    payload = {"scope": scope, "path_or_uri": rel_path, "content": content}
    if force:
        payload["force"] = True
    headers = {"X-API-Key": api_key}

    response = client.post(url, json=payload, headers=headers)

    if response.status_code == 503:
        # Embedding backend down: one retry after Retry-After seconds (or 5
        # if the header is absent), then treat whatever happens next as final.
        delay = 5.0
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                pass
        time.sleep(delay)
        response = client.post(url, json=payload, headers=headers)

    if response.status_code // 100 != 2:
        detail = response.text[:200]
        return f"FAILED {response.status_code} {detail}", 0

    body = response.json()
    outcome = "skipped" if body.get("skipped") else "ok"
    return outcome, int(body.get("passages", 0))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    excludes = resolve_excludes(args.exclude, use_defaults=not args.no_default_excludes)

    try:
        all_md = sorted(root.rglob("*.md"))
        selected = select_files(root, excludes)
    except OSError as exc:
        print(f"error: could not walk --root: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    considered = len(all_md)
    excluded_count = considered - len(selected)

    if args.dry_run:
        for path in selected:
            print(path.relative_to(root).as_posix())
        print(
            f"considered={considered} excluded={excluded_count} "
            f"would_ingest={len(selected)} (dry run, no HTTP call made)"
        )
        return EXIT_OK

    api_key = _resolve_api_key(args)
    if not api_key:
        print(
            "error: no API key. Pass --api-key or set the NESCIO_API_KEY environment variable.",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    ingest_url = f"{args.url.rstrip('/')}/api/v2/ingest"

    ingested = skipped = failed = 0
    total_passages = 0

    with httpx.Client(timeout=args.timeout) as client:
        for path in selected:
            rel_path = path.relative_to(root).as_posix()
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                print(f"{rel_path}: FAILED read-error {exc}", flush=True)
                failed += 1
                continue

            outcome, passages = _ingest_one(
                client, ingest_url, api_key, args.scope, rel_path, content, args.force
            )
            print(f"{rel_path}: {outcome}", flush=True)

            if outcome == "ok":
                ingested += 1
                total_passages += passages
            elif outcome == "skipped":
                skipped += 1
                total_passages += passages
            else:
                failed += 1

    print(
        f"considered={considered} excluded={excluded_count} "
        f"ingested={ingested} skipped={skipped} failed={failed} "
        f"total_passages={total_passages}"
    )
    if args.force and skipped != 0:
        print(f"note: --force was set but skipped={skipped}, expected 0", file=sys.stderr)

    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
