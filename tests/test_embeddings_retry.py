"""Retry of transient embedding failures (issue #71).

Nothing here sleeps or reads a real clock: app.core.embeddings looks up _sleep,
_rand and _clock at call time, so each test replaces them with a recording sleep,
a fixed jitter and a fake clock. _fetch_batch is replaced too, except in the
Retry-After tests, which go through the real HTTP classification with
httpx.MockTransport.
"""
import httpx
import pytest
from pydantic import ValidationError

from app.api.v1 import search as search_module
from app.config import Settings, settings
from app.core import embeddings as embeddings_module
from app.core.embeddings import (
    _fetch,
    get_embeddings,
    get_query_embedding,
    probe_embedding,
)
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.core.retry import RetryBudget, backoff_delay, retry_transient
from tests.factories import make_api_key, make_note, make_passage
from tests.fakes import fake_embedding


class FakeTime:
    """A clock that only moves when the test sleeps or a fake call takes time."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_time(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(embeddings_module, "_sleep", fake.sleep)
    monkeypatch.setattr(embeddings_module, "_clock", fake.clock)
    # Full jitter at its ceiling: the delay equals min(max, base * 2**n), so the
    # expected values below are exact.
    monkeypatch.setattr(embeddings_module, "_rand", lambda: 1.0)
    return fake


@pytest.fixture
def retry_settings(monkeypatch):
    monkeypatch.setattr(settings, "embed_retry_attempts", 3)
    monkeypatch.setattr(settings, "embed_retry_base_delay_s", 0.5)
    monkeypatch.setattr(settings, "embed_retry_max_delay_s", 4.0)
    monkeypatch.setattr(settings, "embed_retry_budget", 2)
    monkeypatch.setattr(settings, "embed_batch_size", 1)
    monkeypatch.setattr(settings, "query_embed_timeout_s", 15.0)


class Script:
    """A _fetch_batch whose n-th call raises or returns what the test scripted.

    Each step is an exception to raise, or a number of seconds the call "takes"
    before succeeding. A failing step may be (exception, seconds) to take time
    first. Records every timeout it was handed.
    """

    def __init__(self, fake_time, *steps):
        self.fake_time = fake_time
        self.steps = list(steps)
        self.timeouts: list[float] = []
        self.calls = 0

    def __call__(self, texts, timeout):
        self.timeouts.append(timeout)
        step = self.steps[self.calls] if self.calls < len(self.steps) else None
        self.calls += 1
        if isinstance(step, tuple):
            step, took = step
            self.fake_time.now += took
        if isinstance(step, Exception):
            raise step
        return [fake_embedding(text) for text in texts]


def _install(monkeypatch, fake_time, *steps):
    script = Script(fake_time, *steps)
    monkeypatch.setattr(embeddings_module, "_fetch_batch", script)
    return script


# --- ingest path -----------------------------------------------------------


def test_a_transient_failure_followed_by_success_returns_the_vectors(
    monkeypatch, fake_time, retry_settings
):
    script = _install(monkeypatch, fake_time, EmbeddingBackendError("blip"))

    vectors = get_embeddings(["alpha"])

    assert vectors == [fake_embedding("alpha")]
    assert script.calls == 2
    assert fake_time.sleeps == [0.5]


def test_ingest_delays_double_up_to_the_cap(monkeypatch, fake_time, retry_settings):
    monkeypatch.setattr(settings, "embed_retry_attempts", 5)
    monkeypatch.setattr(settings, "embed_retry_budget", 10)
    monkeypatch.setattr(settings, "embed_retry_base_delay_s", 1.0)
    monkeypatch.setattr(settings, "embed_retry_max_delay_s", 3.0)
    failures = [EmbeddingBackendError("down")] * 4
    script = _install(monkeypatch, fake_time, *failures)

    get_embeddings(["alpha"])

    assert script.calls == 5
    assert fake_time.sleeps == [1.0, 2.0, 3.0, 3.0]


def test_the_delay_is_jittered_between_zero_and_the_ceiling(
    monkeypatch, fake_time, retry_settings
):
    monkeypatch.setattr(embeddings_module, "_rand", lambda: 0.25)
    _install(monkeypatch, fake_time, EmbeddingBackendError("blip"))

    get_embeddings(["alpha"])

    assert fake_time.sleeps == [0.125]  # 0.25 * min(4.0, 0.5 * 2**0)


def test_attempts_per_sub_batch_are_limited(monkeypatch, fake_time, retry_settings):
    monkeypatch.setattr(settings, "embed_retry_attempts", 2)
    monkeypatch.setattr(settings, "embed_retry_budget", 10)
    script = _install(monkeypatch, fake_time, *[EmbeddingBackendError("down")] * 9)

    with pytest.raises(EmbeddingBackendError) as raised:
        get_embeddings(["alpha"])

    assert type(raised.value) is EmbeddingBackendError
    assert script.calls == 2


def test_the_budget_is_shared_across_sub_batches_and_then_runs_out(
    monkeypatch, fake_time, retry_settings
):
    # 6 sub-batches, each failing once then succeeding, would need 6 retries.
    # The budget of 2 allows the first two and refuses the third.
    steps = []
    for _ in range(6):
        steps += [EmbeddingBackendError("blip"), None]
    script = _install(monkeypatch, fake_time, *steps)

    with pytest.raises(EmbeddingBackendError):
        get_embeddings([f"text {i}" for i in range(6)])

    # batch 0: fail, ok; batch 1: fail, ok; batch 2: fail, refused.
    assert script.calls == 5
    assert len(fake_time.sleeps) == 2


def test_total_retry_sleep_is_bounded_whatever_the_batch_count(
    monkeypatch, fake_time, retry_settings
):
    monkeypatch.setattr(settings, "embed_retry_budget", 3)
    # Every sub-batch fails once before succeeding: 40 of them would sleep 40
    # times without a shared budget.
    steps = []
    for _ in range(40):
        steps += [EmbeddingBackendError("x"), None]
    script = _install(monkeypatch, fake_time, *steps)

    with pytest.raises(EmbeddingBackendError):
        get_embeddings([f"text {i}" for i in range(40)])

    assert len(fake_time.sleeps) == 3
    assert sum(fake_time.sleeps) <= 3 * settings.embed_retry_max_delay_s
    assert script.calls == 7  # three batches of (fail, ok), then one more fail


def test_a_zero_budget_disables_retry(monkeypatch, fake_time, retry_settings):
    monkeypatch.setattr(settings, "embed_retry_budget", 0)
    script = _install(monkeypatch, fake_time, EmbeddingBackendError("blip"))

    with pytest.raises(EmbeddingBackendError):
        get_embeddings(["alpha"])

    assert script.calls == 1
    assert fake_time.sleeps == []


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingBackendBadResponse("garbled"),
        EmbeddingBackendMisconfigured("wrong model"),
    ],
)
@pytest.mark.parametrize("path", ["ingest", "search"])
def test_bad_responses_and_misconfiguration_are_never_retried(
    monkeypatch, fake_time, retry_settings, error, path
):
    script = _install(monkeypatch, fake_time, error)

    with pytest.raises(type(error)):
        if path == "ingest":
            get_embeddings(["alpha"])
        else:
            get_query_embedding("alpha")

    assert script.calls == 1
    assert fake_time.sleeps == []


def test_a_wrong_count_is_a_bad_response_and_is_not_retried(
    monkeypatch, fake_time, retry_settings
):
    calls = []

    def short(texts, timeout):
        calls.append(texts)
        return []

    monkeypatch.setattr(embeddings_module, "_fetch_batch", short)

    with pytest.raises(EmbeddingBackendBadResponse):
        get_embeddings(["alpha"])

    assert len(calls) == 1


def test_the_probe_does_not_retry(monkeypatch, fake_time, retry_settings):
    script = _install(monkeypatch, fake_time, *[EmbeddingBackendError("x")] * 3)

    with pytest.raises(EmbeddingBackendError):
        probe_embedding()
    with pytest.raises(EmbeddingBackendError):
        _fetch("anything")

    assert script.calls == 2
    assert fake_time.sleeps == []


def test_each_retry_is_logged_without_the_texts(
    monkeypatch, fake_time, retry_settings, caplog
):
    secret = "c7e2d0-caller-supplied-text"
    # alembic/env.py's fileConfig() disables every logger that already exists
    # once any database test has migrated (see tests/test_search_degraded_log.py).
    monkeypatch.setattr(embeddings_module.logger, "disabled", False)
    _install(monkeypatch, fake_time, EmbeddingBackendError(f"quoted {secret}"))

    with caplog.at_level("WARNING", logger="app.core.embeddings"):
        get_embeddings([secret])

    (record,) = [r for r in caplog.records if r.name == "app.core.embeddings"]
    assert record.error_class == "EmbeddingBackendError"
    assert record.attempt == 1
    assert record.delay_s == 0.5
    assert secret not in caplog.text


# --- search path -----------------------------------------------------------


def test_search_retries_a_fast_failure_once_inside_the_deadline(
    monkeypatch, fake_time, retry_settings
):
    monkeypatch.setattr(settings, "embed_retry_attempts", 5)
    monkeypatch.setattr(settings, "embed_retry_budget", 10)
    start = fake_time.now
    # Both calls fail fast, taking 1 s each. Only one retry is allowed.
    script = _install(
        monkeypatch,
        fake_time,
        (EmbeddingBackendError("a"), 1.0),
        (EmbeddingBackendError("b"), 1.0),
    )

    with pytest.raises(EmbeddingBackendError):
        get_query_embedding("alpha")

    assert script.calls == 2
    assert fake_time.now - start <= settings.query_embed_timeout_s
    # The retry gets what is left: 15 - 1 s (first call) - 0.5 s (backoff).
    assert script.timeouts == [15.0, 13.5]


def test_search_succeeds_on_the_retry(monkeypatch, fake_time, retry_settings):
    _install(monkeypatch, fake_time, EmbeddingBackendError("blip"))

    assert len(get_query_embedding("alpha")) == settings.embedding_dimension


def test_search_does_not_retry_after_a_timeout_used_the_whole_deadline(
    monkeypatch, fake_time, retry_settings
):
    script = _install(
        monkeypatch, fake_time, (EmbeddingBackendError("timed out"), 15.0)
    )

    with pytest.raises(EmbeddingBackendError):
        get_query_embedding("alpha")

    assert script.calls == 1
    assert fake_time.sleeps == []


def test_search_does_not_retry_when_the_backoff_would_overrun_the_deadline(
    monkeypatch, fake_time, retry_settings
):
    # 14.8 s used, 0.2 s left; the 0.5 s backoff cannot fit.
    script = _install(monkeypatch, fake_time, (EmbeddingBackendError("slow"), 14.8))

    with pytest.raises(EmbeddingBackendError):
        get_query_embedding("alpha")

    assert script.calls == 1


# --- Retry-After through the real HTTP classification ----------------------


def _http(monkeypatch, responses):
    """Answer posts from `responses` (a list of httpx.Response factories)."""
    real_client = httpx.Client
    seen = []

    def handler(request):
        seen.append(request)
        return responses[min(len(seen), len(responses)) - 1]()

    monkeypatch.setattr(
        embeddings_module.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return seen


def _status(code, headers=None):
    return lambda: httpx.Response(code, headers=headers or {})


def _ok(n=1):
    return lambda: httpx.Response(
        200, json={"embeddings": [fake_embedding("alpha")] * n}
    )


@pytest.mark.parametrize("code", [429, 503])
def test_retry_after_is_honoured_within_the_cap(
    monkeypatch, fake_time, retry_settings, code
):
    seen = _http(monkeypatch, [_status(code, {"Retry-After": "2"}), _ok()])

    get_embeddings(["alpha"])

    assert len(seen) == 2
    assert fake_time.sleeps == [2.0]


def test_retry_after_above_the_cap_is_capped(monkeypatch, fake_time, retry_settings):
    _http(monkeypatch, [_status(429, {"Retry-After": "120"}), _ok()])

    get_embeddings(["alpha"])

    assert fake_time.sleeps == [settings.embed_retry_max_delay_s]


@pytest.mark.parametrize("value", ["soon", "Wed, 21 Oct 2026 07:28:00 GMT", "-5", "nan"])
def test_an_unusable_retry_after_falls_back_to_jittered_backoff(
    monkeypatch, fake_time, retry_settings, value
):
    _http(monkeypatch, [_status(429, {"Retry-After": value}), _ok()])

    get_embeddings(["alpha"])

    assert fake_time.sleeps == [0.5]


@pytest.mark.parametrize("code", [408, 500, 502, 504])
def test_other_transient_statuses_retry_with_jitter(
    monkeypatch, fake_time, retry_settings, code
):
    seen = _http(monkeypatch, [_status(code), _ok()])

    get_embeddings(["alpha"])

    assert len(seen) == 2
    assert fake_time.sleeps == [0.5]


@pytest.mark.parametrize("code", [400, 404, 501, 505])
def test_permanent_statuses_are_not_retried(
    monkeypatch, fake_time, retry_settings, code
):
    seen = _http(monkeypatch, [_status(code)])

    with pytest.raises(EmbeddingBackendMisconfigured):
        get_embeddings(["alpha"])

    assert len(seen) == 1


def test_a_connection_error_is_retried(monkeypatch, fake_time, retry_settings):
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"embeddings": [fake_embedding("alpha")]})

    real_client = httpx.Client
    monkeypatch.setattr(
        embeddings_module.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(flaky), **kw),
    )

    assert get_embeddings(["alpha"]) == [fake_embedding("alpha")]
    assert state["n"] == 2


# --- app.core.retry on its own ---------------------------------------------


def test_backoff_delay_is_capped_and_never_overflows():
    assert backoff_delay(0, 0.5, 4.0, lambda: 1.0) == 0.5
    assert backoff_delay(3, 0.5, 4.0, lambda: 1.0) == 4.0
    assert backoff_delay(10_000, 0.5, 4.0, lambda: 1.0) == 4.0
    assert backoff_delay(2, 0.5, 4.0, lambda: 0.0) == 0.0


def test_the_deadline_check_does_not_spend_the_budget():
    budget = RetryBudget(1)
    now = [0.0]

    def fail():
        raise EmbeddingBackendError("x")

    with pytest.raises(EmbeddingBackendError):
        retry_transient(
            fail,
            retryable=lambda exc: True,
            attempts=3,
            base_delay_s=1.0,
            max_delay_s=1.0,
            budget=budget,
            deadline=0.5,
            sleep=lambda s: None,
            rand=lambda: 1.0,
            clock=lambda: now[0],
        )

    assert budget.remaining == 1


# --- settings --------------------------------------------------------------


def _settings(**overrides):
    return Settings(_env_file=None, database_url="postgresql://x/y", **overrides)


def test_the_defaults_are_the_documented_ones():
    s = _settings()
    assert (
        s.embed_retry_attempts,
        s.embed_retry_base_delay_s,
        s.embed_retry_max_delay_s,
        s.embed_retry_budget,
    ) == (3, 0.5, 4.0, 2)


@pytest.mark.parametrize(
    ("overrides", "name"),
    [
        ({"embed_retry_attempts": 0}, "EMBED_RETRY_ATTEMPTS"),
        ({"embed_retry_attempts": 6}, "EMBED_RETRY_ATTEMPTS"),
        ({"embed_retry_budget": -1}, "EMBED_RETRY_BUDGET"),
        ({"embed_retry_budget": 11}, "EMBED_RETRY_BUDGET"),
        ({"embed_retry_max_delay_s": 0}, "EMBED_RETRY_MAX_DELAY_S"),
        ({"embed_retry_max_delay_s": 10.5}, "EMBED_RETRY_MAX_DELAY_S"),
        ({"embed_retry_max_delay_s": float("nan")}, "EMBED_RETRY_MAX_DELAY_S"),
        ({"embed_retry_base_delay_s": 0}, "EMBED_RETRY_BASE_DELAY_S"),
        ({"embed_retry_base_delay_s": float("inf")}, "EMBED_RETRY_BASE_DELAY_S"),
        # base must not exceed max (default max is 4.0)
        ({"embed_retry_base_delay_s": 5.0}, "EMBED_RETRY_BASE_DELAY_S"),
        (
            {"embed_retry_base_delay_s": 3.0, "embed_retry_max_delay_s": 2.0},
            "EMBED_RETRY_BASE_DELAY_S",
        ),
    ],
)
def test_out_of_range_retry_settings_fail_at_startup_naming_the_variable(
    overrides, name
):
    with pytest.raises(ValidationError, match=name):
        _settings(**overrides)


def test_the_bounds_themselves_are_accepted():
    _settings(embed_retry_attempts=1, embed_retry_budget=0)
    _settings(embed_retry_attempts=5, embed_retry_budget=10)
    _settings(embed_retry_base_delay_s=10.0, embed_retry_max_delay_s=10.0)


# --- end to end ------------------------------------------------------------


def test_search_still_degrades_when_the_retry_fails_too(
    client, db_session, monkeypatch, fake_time, retry_settings
):
    key = make_api_key(db_session, "acme")
    make_passage(
        db_session,
        make_note(db_session, "acme"),
        ordinal=0,
        content="The pelicanium isotope decays slowly.",
    )
    monkeypatch.setattr(search_module, "get_query_embedding", get_query_embedding)
    script = _install(monkeypatch, fake_time, *[EmbeddingBackendError("down")] * 5)

    response = client.post(
        "/api/v1/search",
        json={"q": "pelicanium", "top_k": 10},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200, response.text
    assert response.json()["degraded"] is True
    assert script.calls == 2
