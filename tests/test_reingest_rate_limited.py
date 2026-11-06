"""scripts/reingest.py treats a 429 like a 503: one retry, after a bounded delay.

The file is new because tests/test_scripts.py, which covers the 503 path this shares,
is close to its size limit. The transport stub is the same seam as its
_install_transport, without the recording.
"""
import httpx
import pytest

from scripts import reingest


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run reingest over one note against a handler; return (exit, delays, calls)."""
    (tmp_path / "note.md").write_text("# note\n\nbody\n", encoding="utf-8")
    delays: list[float] = []
    monkeypatch.setattr(reingest.time, "sleep", delays.append)
    real_client = httpx.Client

    def _run(responses: list[httpx.Response]):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return responses[min(len(calls) - 1, len(responses) - 1)]

        monkeypatch.setattr(
            reingest.httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                *args, transport=httpx.MockTransport(handler), **kwargs
            ),
        )
        code = reingest.main(["--root", str(tmp_path), "--api-key", "nm_test_key"])
        return code, delays, calls

    return _run


def _rate_limited(retry_after: str | None = "2") -> httpx.Response:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return httpx.Response(429, json={"detail": "Rate limit exceeded"}, headers=headers)


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"skipped": False, "passages": 3})


def test_a_429_is_retried_once_after_retry_after(run, capsys):
    code, delays, calls = run([_rate_limited("2"), _ok()])

    assert delays == [2.0]
    assert len(calls) == 2
    assert "note.md: ok" in capsys.readouterr().out
    assert code == reingest.EXIT_OK


@pytest.mark.parametrize(
    "retry_after, expected",
    [
        pytest.param("86400", reingest.MAX_RETRY_AFTER_S, id="capped"),
        pytest.param(None, 5.0, id="absent"),
        pytest.param("nan", 5.0, id="nonsense"),
    ],
)
def test_the_429_delay_is_capped_and_falls_back(run, retry_after, expected):
    _, delays, _ = run([_rate_limited(retry_after), _ok()])

    assert delays == [expected]


def test_a_second_429_is_final_and_reported_as_failed(run, capsys):
    code, delays, calls = run([_rate_limited("2")])

    stdout = capsys.readouterr().out
    assert delays == [2.0]
    assert len(calls) == 2  # one retry, not a loop
    assert "note.md: FAILED 429" in stdout
    assert "failed=1" in stdout
    assert code == reingest.EXIT_FAILED


def test_a_503_after_a_429_is_final_too(run, capsys):
    code, _, calls = run([_rate_limited("2"), httpx.Response(503, text="backend down")])

    assert len(calls) == 2
    assert "note.md: FAILED 503" in capsys.readouterr().out
    assert code == reingest.EXIT_FAILED
