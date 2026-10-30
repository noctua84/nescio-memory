# scripts/eval_queries.py
"""Golden-query harness: measure retrieval quality against a real corpus.

    uv run python -m scripts.eval_queries --api-key <key>
        [--queries tests/fixtures/golden_queries.yaml] [--url http://localhost:8080]
        [--repo-filter <scope>] [--top-k N] [--json]

Invoked with -m, not as a path: `python scripts/eval_queries.py` puts scripts/
on sys.path instead of the repository root, so `import app`-style imports
(none here, but keep the convention) would fail.

--url must point at an allowed host: localhost, 127.0.0.1 and ::1 by
default, extended via the comma-separated NESCIO_ALLOWED_HOSTS environment
variable. This exists so a crafted --url cannot exfiltrate the X-API-Key
header to an arbitrary host. --queries must resolve to an existing .yaml/
.yml file, so a crafted path cannot make the script read an arbitrary file.

Loads tests/fixtures/golden_queries.yaml (or --queries), POSTs each query to
`{url}/api/v2/search`, and reports hits@top_k against the file's gate. The
gate is "15 of 20 queries hit in the top 5" by default; the recorded score is
the baseline slice 2's RRF work must beat.

Exit codes:
    0  gate passed
    1  gate failed
    2  usage, connection, or validation error
    3  fewer than gate.minimum_queries queries -- not yet populated, not a
       failure; this is the expected state in slice 1 with the 3 examples
       shipped in the fixture.
"""
from __future__ import annotations

import argparse
import json as json_module
import sys
from dataclasses import dataclass

import httpx
import yaml

from scripts._cli_safety import validate_base_url, validate_queries_path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2
EXIT_NOT_POPULATED = 3


class QueriesFileError(ValueError):
    """The golden-queries YAML is malformed."""


@dataclass
class Gate:
    top_k: int
    threshold: int
    minimum_queries: int


@dataclass
class GoldenQuery:
    query: str
    expect_path_or_uri: list[str]
    notes: str | None = None


@dataclass
class GoldenQueries:
    gate: Gate
    queries: list[GoldenQuery]


def load_queries(path: str) -> GoldenQueries:
    """Load and validate the golden-queries YAML. Raises QueriesFileError.

    Validates the path itself first (must resolve to an existing .yaml/.yml
    file) so a crafted --queries cannot make this open() an arbitrary file.
    """
    try:
        safe_path = validate_queries_path(path)
    except ValueError as exc:
        raise QueriesFileError(str(exc)) from exc

    try:
        with open(safe_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise QueriesFileError(f"{path}: not valid YAML ({exc})") from exc

    if not isinstance(raw, dict):
        raise QueriesFileError(f"{path}: top-level content must be a mapping")

    if raw.get("version") != 1:
        raise QueriesFileError(f"{path}: version must be 1, got {raw.get('version')!r}")

    gate_raw = raw.get("gate")
    if not isinstance(gate_raw, dict):
        raise QueriesFileError(f"{path}: 'gate' section missing or not a mapping")
    try:
        gate = Gate(
            top_k=int(gate_raw["top_k"]),
            threshold=int(gate_raw["threshold"]),
            minimum_queries=int(gate_raw["minimum_queries"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QueriesFileError(f"{path}: gate.top_k/threshold/minimum_queries missing or invalid ({exc})") from exc

    raw_queries = raw.get("queries")
    if not isinstance(raw_queries, list):
        raise QueriesFileError(f"{path}: 'queries' must be a list")

    queries: list[GoldenQuery] = []
    for index, entry in enumerate(raw_queries):
        if not isinstance(entry, dict):
            raise QueriesFileError(f"{path}: queries[{index}] must be a mapping")

        query_text = entry.get("query")
        if not isinstance(query_text, str) or not query_text:
            raise QueriesFileError(f"{path}: queries[{index}] needs a non-empty 'query'")

        expect = entry.get("expect_path_or_uri")
        if isinstance(expect, str) and expect:
            expect_list = [expect]
        elif (
            isinstance(expect, list)
            and expect
            and all(isinstance(item, str) and item for item in expect)
        ):
            expect_list = list(expect)
        else:
            raise QueriesFileError(
                f"{path}: queries[{index}] needs a non-empty 'expect_path_or_uri' "
                "string or non-empty list of strings"
            )

        notes = entry.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise QueriesFileError(f"{path}: queries[{index}]['notes'] must be a string")

        queries.append(GoldenQuery(query=query_text, expect_path_or_uri=expect_list, notes=notes))

    return GoldenQueries(gate=gate, queries=queries)


def score(results: list[dict], expected: str | list[str], top_k: int) -> int | None:
    """Return the 1-based rank of the first hit within top_k, or None."""
    expected_set = {expected} if isinstance(expected, str) else set(expected)
    for rank, result in enumerate(results[:top_k], start=1):
        if result.get("path_or_uri") in expected_set:
            return rank
    return None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-key", required=True, help="API key sent as X-API-Key.")
    parser.add_argument(
        "--queries",
        default="tests/fixtures/golden_queries.yaml",
        help="Path to the golden-queries YAML. Must be an existing .yaml/.yml file.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help=(
            "Base URL of the running service. Host must be on the allowlist "
            "(localhost, 127.0.0.1, ::1 by default; extend via the "
            "NESCIO_ALLOWED_HOSTS environment variable, comma-separated)."
        ),
    )
    parser.add_argument("--repo-filter", default=None, help="Optional scope filter forwarded to the search.")
    parser.add_argument("--top-k", type=int, default=None, help="Overrides gate.top_k from the YAML.")
    parser.add_argument("--json", action="store_true", help="Emit a single JSON object instead of text.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        base_url = validate_base_url(args.url)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        golden = load_queries(args.queries)
    except QueriesFileError as exc:
        print(f"invalid golden-queries file: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"could not read {args.queries}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    top_k = args.top_k if args.top_k is not None else golden.gate.top_k
    total = len(golden.queries)

    if total < golden.gate.minimum_queries:
        print(
            f"not yet populated: {total} query(ies), need at least "
            f"{golden.gate.minimum_queries} for the gate to be meaningful"
        )
        print(f"hits@{top_k}: 0/{total} (gate void)")
        if args.json:
            print(json_module.dumps(
                {
                    "populated": False,
                    "top_k": top_k,
                    "hits": 0,
                    "total": total,
                    "minimum_queries": golden.gate.minimum_queries,
                }
            ))
        return EXIT_NOT_POPULATED

    headers = {"X-API-Key": args.api_key}
    hits = 0
    per_query: list[dict] = []

    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
            for golden_query in golden.queries:
                payload: dict = {"q": golden_query.query, "top_k": top_k}
                if args.repo_filter:
                    payload["repo_filter"] = args.repo_filter

                response = client.post("/api/v2/search", json=payload)
                response.raise_for_status()
                results = response.json().get("results", [])

                rank = score(results, golden_query.expect_path_or_uri, top_k)
                if rank is not None:
                    hits += 1
                    print(f"HIT @{rank}  {golden_query.query}")
                else:
                    misses = [r.get("path_or_uri") for r in results[:top_k]]
                    print(f"MISS  {golden_query.query}")
                    print(f"  expected: {golden_query.expect_path_or_uri}")
                    print(f"  got:      {misses}")

                per_query.append({"query": golden_query.query, "rank": rank})
    except httpx.HTTPError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    passed = hits >= golden.gate.threshold
    print(f"hits@{top_k}: {hits}/{total}")
    print("GATE PASS" if passed else "GATE FAIL")

    if args.json:
        print(json_module.dumps(
            {
                "populated": True,
                "top_k": top_k,
                "hits": hits,
                "total": total,
                "threshold": golden.gate.threshold,
                "passed": passed,
                "queries": per_query,
            }
        ))

    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
