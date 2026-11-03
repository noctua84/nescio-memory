"""GET /ready: dependency status, the time bound, and single-flight.

The database tests use the real Postgres container (no DB mocks). Ollama is
mocked at the HTTP layer with httpx.MockTransport, and the hang / concurrency
tests replace the check functions, since "hangs" cannot be produced by a real
dependency on demand.

Nothing here is tenant data: /ready is unauthenticated and returns no
per-client information, so there is no isolation case to test.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import psycopg.errors
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from app.config import Settings, settings
from app.core import readiness
from app.main import app


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _Probes:
    """Fresh probes for one test, plus the gate a hung check waits on."""

    def __init__(self):
        self.clock = _Clock()
        self.gate = threading.Event()
        self.calls = {"database": 0, "embedding_backend": 0}
        self.database = readiness._SingleFlightProbe(
            "database", lambda t: readiness.check_database(t), self.clock
        )
        self.embedding = readiness._SingleFlightProbe(
            "embedding_backend",
            lambda t: readiness.check_embedding_backend(t),
            self.clock,
        )

    def stub(self, monkeypatch, *, database="ok", embedding="ok"):
        """Replace the check functions with counting stubs.

        A value of "hang" blocks on the gate, then returns "ok". A value of
        None leaves that real check function in place.
        """

        def make(name, outcome):
            def check(timeout_s):
                self.calls[name] += 1
                if outcome == "hang":
                    self.gate.wait(timeout=30)
                    return "ok"
                return outcome

            return check

        if database is not None:
            monkeypatch.setattr(
                readiness, "check_database", make("database", database)
            )
        if embedding is not None:
            monkeypatch.setattr(
                readiness,
                "check_embedding_backend",
                make("embedding_backend", embedding),
            )

    def wait_idle(self, probe, timeout=5.0):
        end = time.monotonic() + timeout
        while probe._inflight is not None:
            assert time.monotonic() < end, "worker never finished"
            time.sleep(0.01)


@pytest.fixture
def probes(monkeypatch):
    """Swap in fresh probes, so no test sees another's cached result."""
    fresh = _Probes()
    monkeypatch.setattr(readiness, "DATABASE_PROBE", fresh.database)
    monkeypatch.setattr(readiness, "EMBEDDING_PROBE", fresh.embedding)
    # Short bounds keep the hang tests quick; TTL 0 unless a test opts in.
    monkeypatch.setattr(settings, "ready_db_timeout_s", 0.3)
    monkeypatch.setattr(settings, "ready_embedding_timeout_s", 0.3)
    monkeypatch.setattr(settings, "ready_cache_ttl_s", 0.0)
    monkeypatch.setattr(readiness.logger, "disabled", False)
    yield fresh
    # Release any hung worker before joining it.
    fresh.gate.set()
    fresh.database.shutdown()
    fresh.embedding.shutdown()


@pytest.fixture
def ready_client():
    # Deliberately not the conftest `client`: that wires a database session and
    # an API key world, and /ready needs neither.
    return TestClient(app)


def _body(response):
    return response.json()


def _statuses(body):
    return (
        body["status"],
        body["checks"]["database"]["status"],
        body["checks"]["embedding_backend"]["status"],
    )


# --- database ---------------------------------------------------------------


def test_a_reachable_database_is_ok(probes, ready_client, engine, monkeypatch):
    monkeypatch.setattr(readiness, "get_engine", lambda: engine)
    # The local backend is in-process, so nothing remote is contacted.
    monkeypatch.setattr(settings, "embedding_backend", "local")

    response = ready_client.get("/ready")

    assert response.status_code == 200
    assert _statuses(_body(response)) == ("ready", "ok", "not_checked")


def test_an_unreachable_database_is_503_not_ready(
    probes, ready_client, monkeypatch, caplog
):
    # conftest's placeholder DATABASE_URL points at localhost:1, a closed port.
    # A dedicated engine rather than get_engine(), so the cached app engine is
    # not created as a side effect.
    dead = create_engine(settings.database_url, connect_args={"connect_timeout": 2})
    monkeypatch.setattr(readiness, "get_engine", lambda: dead)
    monkeypatch.setattr(settings, "embedding_backend", "local")
    # Refusal is instant on Linux but not on Windows, where connecting to a
    # closed port burns the whole connect_timeout (libpq's floor is 2 s per
    # address). The bound must outlast that, or this would test the timeout
    # path instead of the failure path.
    monkeypatch.setattr(settings, "ready_db_timeout_s", 10.0)

    with caplog.at_level(logging.WARNING, logger=readiness.logger.name):
        response = ready_client.get("/ready")

    assert response.status_code == 503
    assert _statuses(_body(response)) == ("not_ready", "fail", "not_checked")
    # The body carries no exception text or host name ...
    assert "localhost" not in response.text
    assert "placeholder" not in response.text
    # ... the log does carry the detail.
    assert any(
        "dependency=database" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )
    dead.dispose()


def test_a_cancelled_statement_is_reported_as_timeout(monkeypatch):
    class Engine:
        def connect(self):
            raise OperationalError("SELECT 1", {}, psycopg.errors.QueryCanceled())

    monkeypatch.setattr(readiness, "get_engine", lambda: Engine())

    assert readiness.check_database(0.5) == "timeout"


# --- embedding backend --------------------------------------------------------


def _ollama(monkeypatch, handler):
    seen = []

    def recording(request):
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording)
    # Bound before patching, as in tests/test_scripts.py: the patch rebinds the
    # attribute on the shared httpx module, so the factory must not look it up.
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(readiness.httpx, "Client", factory)
    return seen


def _refused(request):
    raise httpx.ConnectError("refused", request=request)


def _timed_out(request):
    raise httpx.ReadTimeout("slow", request=request)


def test_a_pulled_model_is_ok_and_the_probe_hits_api_show(
    probes, ready_client, monkeypatch
):
    probes.stub(monkeypatch, embedding=None)  # database ok, real Ollama check
    monkeypatch.setattr(settings, "embedding_backend", "ollama")
    monkeypatch.setattr(settings, "ollama_url", "http://ollama.test/proxy/api/embed")
    monkeypatch.setattr(settings, "ollama_model", "some-model:1b")
    seen = _ollama(monkeypatch, lambda r: httpx.Response(200, json={"details": {}}))

    response = ready_client.get("/ready")

    assert response.status_code == 200
    assert _statuses(_body(response)) == ("ready", "ok", "ok")
    assert len(seen) == 1
    # The validated /api/embed suffix is stripped, a reverse-proxy prefix kept.
    assert str(seen[0].url) == "http://ollama.test/proxy/api/show"
    assert seen[0].method == "POST"
    assert seen[0].read() == b'{"model":"some-model:1b"}'


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (lambda r: httpx.Response(404, json={"error": "model not found"}), "fail"),
        (lambda r: httpx.Response(500), "fail"),
        (_refused, "fail"),
        (_timed_out, "timeout"),
    ],
    ids=["model-not-pulled", "server-error", "connection-error", "timeout"],
)
def test_an_ollama_failure_is_200_degraded(
    probes, ready_client, monkeypatch, handler, expected, caplog
):
    probes.stub(monkeypatch, embedding=None)  # database ok, real Ollama check
    monkeypatch.setattr(settings, "embedding_backend", "ollama")
    monkeypatch.setattr(settings, "ollama_url", "http://secret-host.test/api/embed")
    _ollama(monkeypatch, handler)

    with caplog.at_level(logging.WARNING, logger=readiness.logger.name):
        response = ready_client.get("/ready")

    assert response.status_code == 200
    assert _statuses(_body(response)) == ("degraded", "ok", expected)
    assert "secret-host" not in response.text
    assert any("dependency=embedding_backend" in r.getMessage() for r in caplog.records)


def test_the_database_failing_wins_over_a_healthy_ollama(
    probes, ready_client, monkeypatch
):
    probes.stub(monkeypatch, database="fail", embedding="ok")

    response = ready_client.get("/ready")

    assert response.status_code == 503
    assert _statuses(_body(response)) == ("not_ready", "fail", "ok")


def test_both_failing_is_still_503(probes, ready_client, monkeypatch):
    probes.stub(monkeypatch, database="fail", embedding="fail")

    response = ready_client.get("/ready")

    assert response.status_code == 503
    assert _statuses(_body(response)) == ("not_ready", "fail", "fail")


def test_a_check_that_raises_is_reported_as_fail_and_does_not_wedge(
    probes, ready_client, monkeypatch
):
    def boom(timeout_s):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(readiness, "check_database", boom)
    monkeypatch.setattr(readiness, "check_embedding_backend", lambda t: "ok")

    first = ready_client.get("/ready")
    probes.wait_idle(probes.database)
    second = ready_client.get("/ready")

    assert first.status_code == second.status_code == 503
    assert "internal detail" not in first.text


# --- time bound and single flight --------------------------------------------


def test_a_hanging_check_returns_within_the_timeout(probes, ready_client, monkeypatch):
    probes.stub(monkeypatch, database="hang")

    started = time.monotonic()
    response = ready_client.get("/ready")
    elapsed = time.monotonic() - started

    assert response.status_code == 503
    assert _statuses(_body(response)) == ("not_ready", "timeout", "ok")
    assert 0.25 <= elapsed < 0.3 + 1.0


def test_both_dependencies_hanging_cost_the_larger_timeout_not_the_sum(
    probes, ready_client, monkeypatch
):
    probes.stub(monkeypatch, database="hang", embedding="hang")
    monkeypatch.setattr(settings, "ready_db_timeout_s", 0.5)
    monkeypatch.setattr(settings, "ready_embedding_timeout_s", 0.5)

    started = time.monotonic()
    response = ready_client.get("/ready")
    elapsed = time.monotonic() - started

    assert _statuses(_body(response)) == ("not_ready", "timeout", "timeout")
    assert elapsed < 0.5 + 0.4, f"waits were sequential: {elapsed:.2f}s"


def test_concurrent_requests_while_hanging_run_exactly_one_check(
    probes, ready_client, monkeypatch
):
    probes.stub(monkeypatch, database="hang")
    # Long enough that all ten are in flight at once.
    monkeypatch.setattr(settings, "ready_db_timeout_s", 1.0)

    with ThreadPoolExecutor(max_workers=10) as pool:
        responses = list(pool.map(lambda _: ready_client.get("/ready"), range(10)))

    assert [r.status_code for r in responses] == [503] * 10
    assert {_statuses(r.json())[1] for r in responses} == {"timeout"}
    assert probes.calls["database"] == 1

    # A later request while the same check is still hanging submits nothing.
    again = ready_client.get("/ready")
    assert _statuses(again.json())[1] == "timeout"
    assert probes.calls["database"] == 1

    # Once the hung check finishes, the next request may start a fresh one.
    probes.gate.set()
    probes.wait_idle(probes.database)
    after = ready_client.get("/ready")
    assert after.status_code == 200
    assert probes.calls["database"] == 2


# --- TTL ---------------------------------------------------------------------


def test_a_result_is_reused_within_the_ttl_then_refreshed(
    probes, ready_client, monkeypatch
):
    monkeypatch.setattr(settings, "ready_cache_ttl_s", 5.0)
    probes.stub(monkeypatch)

    ready_client.get("/ready")
    probes.clock.now += 4.9
    ready_client.get("/ready")
    assert probes.calls == {"database": 1, "embedding_backend": 1}

    probes.clock.now += 0.2  # 5.1 s since the check finished
    ready_client.get("/ready")
    assert probes.calls == {"database": 2, "embedding_backend": 2}


def test_a_ttl_of_zero_checks_every_request(probes, ready_client, monkeypatch):
    probes.stub(monkeypatch)

    ready_client.get("/ready")
    ready_client.get("/ready")

    assert probes.calls == {"database": 2, "embedding_backend": 2}


def test_a_cached_failure_is_served_until_the_ttl_ends(
    probes, ready_client, monkeypatch
):
    monkeypatch.setattr(settings, "ready_cache_ttl_s", 5.0)
    probes.stub(monkeypatch, database="fail")
    assert ready_client.get("/ready").status_code == 503

    probes.stub(monkeypatch, database="ok")
    assert ready_client.get("/ready").status_code == 503  # still the cached fail

    probes.clock.now += 6
    assert ready_client.get("/ready").status_code == 200


# --- wiring ------------------------------------------------------------------


def test_ready_needs_no_api_key(probes, ready_client, monkeypatch):
    probes.stub(monkeypatch)

    response = ready_client.get("/ready")  # no X-API-Key header

    assert response.status_code == 200


def test_health_contacts_nothing(probes, ready_client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("/health contacted a dependency")

    monkeypatch.setattr(readiness, "check_database", forbidden)
    monkeypatch.setattr(readiness, "check_embedding_backend", forbidden)
    monkeypatch.setattr(readiness, "get_engine", forbidden)
    monkeypatch.setattr(readiness.httpx, "Client", forbidden)

    response = ready_client.get("/health")

    assert response.status_code == 200
    assert probes.database._inflight is None
    assert probes.embedding._inflight is None


def test_the_openapi_spec_documents_ready():
    operation = app.openapi()["paths"]["/ready"]["get"]

    assert operation["operationId"] == "readiness"
    assert operation["tags"] == ["Ops"]
    assert operation["description"]
    assert "503" in operation["responses"]
    assert "security" not in operation


# --- settings ----------------------------------------------------------------


def _settings(**overrides):
    base = {
        "database_url": "postgresql+psycopg://u:p@localhost:5432/db",
        "ollama_url": "http://localhost:11434/api/embed",
    }
    base.update(overrides)
    return Settings(**base)


def test_the_readiness_defaults():
    assert Settings.model_fields["ready_db_timeout_s"].default == 2.0
    assert Settings.model_fields["ready_embedding_timeout_s"].default == 2.0
    assert Settings.model_fields["ready_cache_ttl_s"].default == 5.0


@pytest.mark.parametrize(
    ("field", "env_var"),
    [
        ("ready_db_timeout_s", "READY_DB_TIMEOUT_S"),
        ("ready_embedding_timeout_s", "READY_EMBEDDING_TIMEOUT_S"),
    ],
)
@pytest.mark.parametrize("bad", [0, -1, 10.5, float("nan"), float("inf")])
def test_a_bad_ready_timeout_fails_naming_the_env_var(field, env_var, bad):
    with pytest.raises(ValueError, match=env_var):
        _settings(**{field: bad})


@pytest.mark.parametrize("good", [0.1, 2, 10])
def test_ready_timeouts_at_the_bounds_are_accepted(good):
    assert _settings(ready_db_timeout_s=good, ready_embedding_timeout_s=good)


@pytest.mark.parametrize("bad", [-0.1, 60.5, float("nan")])
def test_a_bad_ready_cache_ttl_fails_naming_the_env_var(bad):
    with pytest.raises(ValueError, match="READY_CACHE_TTL_S"):
        _settings(ready_cache_ttl_s=bad)


@pytest.mark.parametrize("good", [0, 5, 60])
def test_ready_cache_ttl_at_the_bounds_is_accepted(good):
    assert _settings(ready_cache_ttl_s=good)
