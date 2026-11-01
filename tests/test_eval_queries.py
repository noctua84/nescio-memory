"""scripts/eval_queries.py refuses to score a degraded measurement.

When the server cannot embed a query it answers ``"degraded": true`` with
FTS-only results. Such a response is not a hybrid result, so any degraded
response invalidates the whole run: exit 2, a count-only message on stderr
(never the query text), no ``hits@``/``GATE`` verdict on stdout, and with
``--json`` a payload marked ``"valid": false`` / ``"passed": null``.

No network and no database: every request goes through an
``httpx.MockTransport`` and the queries file is synthetic, written to tmp_path.
"""
import json

import httpx
import pytest

from scripts import eval_queries

URL = "http://127.0.0.1:8080"

# Synthetic queries only -- this is a public repository.
QUERIES = {
    "synthetic alpha lookup": "notes/alpha.md",
    "synthetic beta lookup": "notes/beta.md",
    "synthetic gamma lookup": "notes/gamma.md",
}

YAML = """version: 1
gate:
  top_k: 5
  threshold: 2
  minimum_queries: 3
queries:
  - query: synthetic alpha lookup
    expect_path_or_uri: notes/alpha.md
  - query: synthetic beta lookup
    expect_path_or_uri: notes/beta.md
  - query: synthetic gamma lookup
    expect_path_or_uri: notes/gamma.md
"""

_ABSENT = object()


@pytest.fixture
def queries_file(tmp_path, monkeypatch):
    monkeypatch.delenv("NESCIO_ALLOWED_HOSTS", raising=False)
    path = tmp_path / "golden.yaml"
    path.write_text(YAML, encoding="utf-8")
    return str(path)


def _run(monkeypatch, queries_file, degraded_by_query, *, hit=True, extra=()):
    """Run main() with each query answered per ``degraded_by_query``.

    ``degraded_by_query`` maps query text to the response's ``degraded`` value;
    ``_ABSENT`` omits the key entirely. ``hit`` controls whether the expected
    path is in the results. Returns (exit_code, list of request bodies).
    """
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/search"
        body = json.loads(request.content)
        seen.append(body)
        expected = QUERIES[body["q"]]
        results = [{"path_or_uri": expected if hit else "notes/unrelated.md"}]
        response: dict = {"results": results}
        flag = degraded_by_query[body["q"]]
        if flag is not _ABSENT:
            response["degraded"] = flag
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(eval_queries.httpx, "Client", factory)
    code = eval_queries.main(
        ["--api-key", "test-key", "--queries", queries_file, "--url", URL, *extra]
    )
    return code, seen


def _flags(alpha, beta, gamma) -> dict:
    return dict(zip(QUERIES, (alpha, beta, gamma)))


def _json_payload(out: str) -> dict:
    return json.loads(out.strip().splitlines()[-1])


def _assert_no_query_text(text: str) -> None:
    for query in QUERIES:
        assert query not in text
    assert "synthetic" not in text


# --- degraded runs ---------------------------------------------------------


def test_one_degraded_response_is_an_error_without_verdict(monkeypatch, queries_file, capsys):
    code, seen = _run(monkeypatch, queries_file, _flags(False, True, False))
    cap = capsys.readouterr()

    assert code == eval_queries.EXIT_ERROR == 2
    assert cap.err.strip() == (
        "error: 1 of 3 responses were degraded (FTS-only fallback); "
        "the measurement is not valid"
    )
    _assert_no_query_text(cap.err)
    assert "GATE" not in cap.out
    assert "hits@" not in cap.out
    # The run still queries every entry so the count is complete.
    assert [b["q"] for b in seen] == list(QUERIES)


def test_degraded_overrides_a_passing_score(monkeypatch, queries_file, capsys):
    # All three would hit (3 >= threshold 2); the degraded one still voids it.
    code, _ = _run(monkeypatch, queries_file, _flags(True, False, False), hit=True)
    cap = capsys.readouterr()

    assert code == 2
    assert "GATE PASS" not in cap.out
    assert "error: 1 of 3 responses were degraded" in cap.err


def test_degraded_overrides_a_failing_score(monkeypatch, queries_file, capsys):
    code, _ = _run(monkeypatch, queries_file, _flags(False, False, True), hit=False)
    cap = capsys.readouterr()

    assert code == 2
    assert "GATE FAIL" not in cap.out
    assert "error: 1 of 3 responses were degraded" in cap.err


def test_one_degraded_json_is_invalid_with_null_verdict(monkeypatch, queries_file, capsys):
    code, _ = _run(monkeypatch, queries_file, _flags(False, True, False), extra=["--json"])
    cap = capsys.readouterr()

    assert code == 2
    payload = _json_payload(cap.out)
    assert payload["valid"] is False
    assert payload["passed"] is None
    assert payload["degraded"] == 1
    assert payload["total"] == 3
    assert "GATE" not in cap.out
    assert "hits@" not in cap.out
    assert cap.err.strip() == (
        "error: 1 of 3 responses were degraded (FTS-only fallback); "
        "the measurement is not valid"
    )
    _assert_no_query_text(cap.err)


def test_all_degraded_counts_every_response(monkeypatch, queries_file, capsys):
    code, _ = _run(monkeypatch, queries_file, _flags(True, True, True), extra=["--json"])
    cap = capsys.readouterr()

    assert code == 2
    payload = _json_payload(cap.out)
    assert payload["degraded"] == 3
    assert payload["valid"] is False
    assert payload["passed"] is None
    assert cap.err.strip() == (
        "error: 3 of 3 responses were degraded (FTS-only fallback); "
        "the measurement is not valid"
    )


def test_all_degraded_text_mode(monkeypatch, queries_file, capsys):
    code, _ = _run(monkeypatch, queries_file, _flags(True, True, True))
    cap = capsys.readouterr()

    assert code == 2
    assert "GATE" not in cap.out
    assert "hits@" not in cap.out
    assert cap.err.strip().startswith("error: 3 of 3 responses were degraded")
    _assert_no_query_text(cap.err)


# --- non-degraded runs -----------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param((False, False, False), id="all-false"),
        pytest.param((None, None, None), id="all-null"),
        pytest.param((_ABSENT, _ABSENT, _ABSENT), id="all-absent"),
        pytest.param((False, None, _ABSENT), id="mixed"),
    ],
)
def test_non_degraded_values_are_not_counted(monkeypatch, queries_file, capsys, flags):
    code, _ = _run(monkeypatch, queries_file, _flags(*flags), extra=["--json"])
    cap = capsys.readouterr()

    assert code == eval_queries.EXIT_PASS == 0
    assert cap.err == ""
    assert "hits@5: 3/3" in cap.out
    assert "GATE PASS" in cap.out
    payload = _json_payload(cap.out)
    assert payload["valid"] is True
    assert payload["degraded"] == 0
    assert payload["passed"] is True
    assert payload["hits"] == 3


def test_non_degraded_text_mode_passes(monkeypatch, queries_file, capsys):
    code, _ = _run(monkeypatch, queries_file, _flags(False, _ABSENT, None))
    cap = capsys.readouterr()

    assert code == 0
    assert cap.err == ""
    assert "hits@5: 3/3" in cap.out
    assert "GATE PASS" in cap.out


def test_non_degraded_failing_run_keeps_fail_exit(monkeypatch, queries_file, capsys):
    code, _ = _run(monkeypatch, queries_file, _flags(False, None, _ABSENT), hit=False, extra=["--json"])
    cap = capsys.readouterr()

    assert code == eval_queries.EXIT_FAIL == 1
    assert cap.err == ""
    assert "hits@5: 0/3" in cap.out
    assert "GATE FAIL" in cap.out
    payload = _json_payload(cap.out)
    assert payload["valid"] is True
    assert payload["degraded"] == 0
    assert payload["passed"] is False
    assert payload["hits"] == 0


def test_non_degraded_failing_run_text_mode(monkeypatch, queries_file, capsys):
    code, _ = _run(monkeypatch, queries_file, _flags(False, False, False), hit=False)
    cap = capsys.readouterr()

    assert code == 1
    assert "GATE FAIL" in cap.out
    assert "degraded" not in cap.err
