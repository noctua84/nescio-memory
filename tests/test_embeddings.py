"""Unit tests for get_embedding's failure translation.

No database and no container: these call get_embedding directly with a stubbed
transport, so a failure here points at the embedding module rather than at the
HTTP layer.
"""
import math

import httpx
import pytest

from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.embeddings import FLOAT32_MAX, get_embedding
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)


class _StubClient:
    """Stands in for httpx.Client. `behaviour` decides what post() does."""

    def __init__(self, behaviour):
        self._behaviour = behaviour

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self._behaviour()


def _install(monkeypatch, behaviour):
    monkeypatch.setattr(embeddings_module.httpx, "Client", _StubClient(behaviour))


def _response(payload, status_code=200):
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "http://ollama.invalid/api/embeddings"),
    )


def _raw_response(body: bytes, status_code=200):
    """A response whose body is handed over verbatim, bypassing httpx's encoder.

    Sibling of `_response()` rather than a change to it, because `_response()`
    structurally cannot carry the cases below: httpx 0.28's `json=` kwarg
    serializes with `allow_nan=False`, so a NaN or Infinity in the payload dies
    with `ValueError: Out of range float values are not JSON compliant` before
    the stubbed transport is ever reached.

    That asymmetry is precisely the production hole. httpx refuses to *emit*
    these values, but `response.json()` decodes with stdlib `json`, which
    accepts the non-standard bare `NaN` / `Infinity` / `-Infinity` literals by
    default and yields real non-finite floats. So a backend that writes its JSON
    by hand -- or any non-Python one -- can put them on the wire, and this
    service will parse them without complaint. Writing the payload as wire bytes
    is the only way to reproduce that, and it makes these tests a regression
    guard on the real path rather than a restatement of `math.isfinite`.
    """
    return httpx.Response(
        status_code=status_code,
        content=body,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "http://ollama.invalid/api/embeddings"),
    )


def _embedding_body(literal: str, index: int = 0) -> bytes:
    """An Ollama-shaped body with `literal` written at `index`, hand-rolled.

    Sized from settings for the same reason the full-length cases below are: at
    any other length the dimension check fires first and the case stops testing
    what it is named for.
    """
    components = ["0.01"] * settings.embedding_dimension
    components[index] = literal
    return b'{"embedding": [' + ", ".join(components).encode("ascii") + b"]}"


NON_FINITE_LITERALS = ["NaN", "Infinity", "-Infinity"]

# Index 0 and the last index. A loop that examined only the first component --
# or that broke out early after a reordering -- would still catch every
# index-0 case while missing the realistic one, where a backend's degenerate
# component sits somewhere in the middle of a 1024-wide vector.
NON_FINITE_POSITIONS = [0, settings.embedding_dimension - 1]

# The wording _validated() uses for a component it can evaluate as a float and
# still will not accept -- non-finite, or finite but outside float32's range.
# Matched on rather than left implicit because _fetch() translates a JSON
# *decode* failure into the very same exception class, so a body that never
# parsed at all would satisfy a bare pytest.raises(EmbeddingBackendBadResponse)
# with the component loop never entered. Pinning the message is what makes the
# rejection tests statements about the range check rather than about any of the
# half-dozen other ways this call can fail.
UNUSABLE_COMPONENT_MESSAGE = "expected finite float32 values"

# And the wording for the one component _validated() cannot even evaluate: an
# integer literal too large for float conversion, where math.isfinite() raises
# instead of returning False.
UNEVALUABLE_COMPONENT_MESSAGE = "too large to evaluate as a float"


def test_connection_failure_raises_embedding_backend_error(monkeypatch):
    def refuse():
        raise httpx.ConnectError("connection refused")

    _install(monkeypatch, refuse)
    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Exact type, not isinstance: pytest.raises matches subclasses, so asserting
    # the base class alone would not catch a regression that raised
    # EmbeddingBackendBadResponse for a connection failure.
    assert type(excinfo.value) is EmbeddingBackendError


def test_timeout_raises_embedding_backend_error(monkeypatch):
    def time_out():
        raise httpx.ReadTimeout("too slow")

    _install(monkeypatch, time_out)
    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Exact type, not isinstance: pytest.raises matches subclasses, so asserting
    # the base class alone would not catch a regression that raised
    # EmbeddingBackendBadResponse for a timeout.
    assert type(excinfo.value) is EmbeddingBackendError


def test_non_2xx_raises_embedding_backend_error(monkeypatch):
    _install(monkeypatch, lambda: _response({"error": "boom"}, status_code=500))
    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Exact type, not isinstance: pytest.raises matches subclasses, so asserting
    # the base class alone would not catch a regression that raised
    # EmbeddingBackendBadResponse for a non-2xx response.
    assert type(excinfo.value) is EmbeddingBackendError


def test_missing_embedding_key_raises_bad_response(monkeypatch):
    _install(monkeypatch, lambda: _response({"not_what_we_expected": []}))
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embedding("anything")


def test_non_json_body_raises_bad_response(monkeypatch):
    def html_page():
        return httpx.Response(
            status_code=200,
            text="<html>proxy error</html>",
            request=httpx.Request("POST", "http://ollama.invalid/api/embeddings"),
        )

    _install(monkeypatch, html_page)
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embedding("anything")


def test_bad_response_is_a_subclass_so_one_handler_covers_both(monkeypatch):
    # A caller that does not care which kind of failure occurred should be able
    # to catch the base class alone.
    assert issubclass(EmbeddingBackendBadResponse, EmbeddingBackendError)


def test_a_successful_call_still_returns_the_vector(monkeypatch):
    vector = [0.01] * settings.embedding_dimension
    _install(monkeypatch, lambda: _response({"embedding": vector}))
    assert get_embedding("anything") == vector


def test_a_malformed_ollama_url_is_reported_as_misconfiguration(monkeypatch):
    def invalid_url():
        raise httpx.InvalidURL("no scheme supplied")

    _install(monkeypatch, invalid_url)
    with pytest.raises(EmbeddingBackendMisconfigured):
        get_embedding("anything")


def test_a_local_backend_runtime_failure_is_treated_as_transient(monkeypatch):
    class _OutOfMemory:
        def encode(self, text):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(embeddings_module.settings, "embedding_backend", "local")
    monkeypatch.setattr(embeddings_module, "_get_local_model", lambda: _OutOfMemory())

    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Transient, not misconfiguration: a later attempt may succeed, so this must
    # map to 503 rather than 500.
    assert type(excinfo.value) is EmbeddingBackendError


def test_a_missing_local_extra_stays_a_misconfiguration(monkeypatch):
    def missing_extra():
        raise EmbeddingBackendMisconfigured("local-embeddings extra not installed")

    monkeypatch.setattr(embeddings_module.settings, "embedding_backend", "local")
    monkeypatch.setattr(embeddings_module, "_get_local_model", missing_extra)

    # Guards the re-raise clause: without it, the broad `except Exception` would
    # downgrade this permanent fault to a transient 503.
    with pytest.raises(EmbeddingBackendMisconfigured):
        get_embedding("anything")


@pytest.mark.parametrize(
    "bad_value, why",
    [
        ([], "empty"),
        ([0.1] * (settings.embedding_dimension + 1), "wrong dimension"),
        ("notavector", "a string"),
        (None, "null"),
        ([None] * settings.embedding_dimension, "a list of nulls"),
        ({"a": 1}, "an object"),
        ([True] * settings.embedding_dimension, "booleans"),
    ],
)
def test_an_unusable_embedding_value_is_rejected(monkeypatch, bad_value, why):
    # The key is present in every case, so the key-existence check passes and
    # only the shape check can catch these.
    #
    # The two full-length cases are sized from settings rather than written as
    # a literal on purpose: at any other length they fail on the dimension
    # check before the component-type check is ever reached, and the case stops
    # testing what it is named for.
    _install(monkeypatch, lambda: _response({"embedding": bad_value}))
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embedding(f"a response carrying {why}")


def test_a_correctly_shaped_embedding_still_passes(monkeypatch):
    vector = [0.01] * settings.embedding_dimension
    _install(monkeypatch, lambda: _response({"embedding": vector}))
    assert get_embedding("anything") == vector


@pytest.mark.parametrize("literal", NON_FINITE_LITERALS)
@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_a_non_finite_embedding_component_is_rejected(monkeypatch, literal, index):
    # These are type-correct and correctly sized: isinstance(float("nan"), float)
    # is True and the vector is exactly EMBEDDING_DIMENSION wide, so neither of
    # the older checks can catch them and only the finiteness check can.
    #
    # Without it they reach pgvector, which rejects them with SQLSTATE 22000 on
    # both insert and distance comparison -- and since DataError now maps to 400,
    # a broken backend would make every request look like a client error while
    # 5xx alerting stayed green. 503 is the honest answer: the backend returned
    # something unusable and it is not the caller's fault.
    #
    # The `match=` is what keeps this honest. _fetch() raises the same
    # EmbeddingBackendBadResponse for a body that failed to parse, so a typo in
    # _embedding_body() -- or a stdlib json that stopped accepting the bare
    # literals -- would turn a bare pytest.raises green while the finiteness
    # check was never reached. The premise test below pins the same thing from
    # the other side; both are cheap and they fail differently, which is useful.
    _install(monkeypatch, lambda: _raw_response(_embedding_body(literal, index)))
    with pytest.raises(EmbeddingBackendBadResponse, match=UNUSABLE_COMPONENT_MESSAGE):
        get_embedding(f"a response carrying {literal} at index {index}")


@pytest.mark.parametrize("literal", NON_FINITE_LITERALS)
@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_the_raw_payload_really_decodes_to_a_non_finite_float(literal, index):
    """The premise the test above rests on, pinned so it cannot pass vacuously.

    _fetch() translates a JSON decode failure into the same
    EmbeddingBackendBadResponse that _validated() raises for a non-finite
    component. So if stdlib json ever stopped accepting the bare literals -- or
    if _embedding_body() produced something malformed -- the rejection test
    above would still go green, for entirely the wrong reason, with the
    finiteness check never reached at all. This asserts the body decodes, that
    it decodes to the right width, and that the component in question really is
    a non-finite float by the time _validated() sees it.
    """
    vector = _raw_response(_embedding_body(literal, index)).json()["embedding"]
    assert len(vector) == settings.embedding_dimension
    assert isinstance(vector[index], float)
    assert not math.isfinite(vector[index])


def test_a_finite_vector_delivered_as_raw_bytes_still_passes(monkeypatch):
    # The control for the helper itself. _raw_response() hand-builds its JSON, so
    # a mistake in it would reject everything -- including the cases above, which
    # would then be measuring nothing. Same transport, same construction, one
    # finite value substituted.
    _install(monkeypatch, lambda: _raw_response(_embedding_body("0.5")))
    expected = [0.01] * settings.embedding_dimension
    expected[0] = 0.5
    assert get_embedding("anything") == expected


def test_a_full_length_vector_of_ints_still_passes(monkeypatch):
    # math.isfinite accepts ints, so the finiteness check composes with the type
    # check's existing int acceptance. Pinned because a reordering that ran
    # isfinite before the type check, or one that narrowed it to floats, would
    # break integer vectors while every rejection case above stayed green.
    vector = [0] * settings.embedding_dimension
    _install(monkeypatch, lambda: _response({"embedding": vector}))
    assert get_embedding("anything") == vector


def _vector_with(value, index: int = 0) -> list[float]:
    """A full-length vector of ordinary components with `value` at `index`.

    The Python-side sibling of `_embedding_body()`, used for the cases that are
    perfectly ordinary JSON numbers and therefore need no raw-bytes workaround.
    Sized from settings for the same reason every other full-length case here
    is: at any other length the dimension check fires first and the case stops
    testing what it is named for.
    """
    vector = [0.01] * settings.embedding_dimension
    vector[index] = value
    return vector


# Finite in Python's float64 and unrepresentable in pgvector's float4. 1e39 is
# barely over the bound, 1e300 is absurdly over it, and -1e39 covers the
# negative side -- the guard compares abs(), so a version that only checked the
# positive end would pass the first two and fail the third.
FLOAT32_OVERFLOW_VALUES = [1e39, 1e300, -1e39]


@pytest.mark.parametrize("value", FLOAT32_OVERFLOW_VALUES)
@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_a_float32_overflowing_component_is_rejected(monkeypatch, value, index):
    # math.isfinite(1e39) is True, so finiteness alone does not catch this: the
    # value is a perfectly well-behaved float64 and a perfectly well-formed JSON
    # number. The destination column is float4, though, and pgvector rejects it
    # with SQLSTATE 22003 ("out of range for type vector") on insert *and* on
    # distance comparison. Since DataError maps to 400, an unguarded 1e39 turns
    # a wholly broken backend into a stream of *client* errors while 5xx alerts
    # and SLOs stay green -- the same inversion the NaN case guards against,
    # reached by a value that needs no non-standard JSON at all.
    #
    # Driven through `_response()`/`json=` deliberately: these values survive
    # httpx's encoder, so this exercises the ordinary decode path rather than
    # the raw-bytes workaround the non-finite cases are forced into.
    _install(monkeypatch, lambda: _response({"embedding": _vector_with(value, index)}))
    with pytest.raises(EmbeddingBackendBadResponse, match=UNUSABLE_COMPONENT_MESSAGE):
        get_embedding(f"a response carrying {value} at index {index}")


@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_the_largest_float32_is_accepted_rather_than_rejected(monkeypatch, index):
    """The bound's inclusive side, which nothing else in this file would catch.

    Every other case here asserts a rejection, so an off-by-one bound -- `>=`
    where the implementation means `>` -- would leave the whole file green while
    the service refused a legitimate component that round-trips through float32
    exactly. Paired with the just-over case below, this pins the boundary from
    both sides.

    FLOAT32_MAX is imported from the module rather than written out as a literal
    so the test tracks the constant. A future correction to the bound should
    move this test with it; a re-typed literal would instead start failing for a
    reason that looks like a bug in the implementation.
    """
    expected = _vector_with(FLOAT32_MAX, index)
    _install(monkeypatch, lambda: _response({"embedding": expected}))
    assert get_embedding("anything") == expected


@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_a_value_just_past_the_largest_float32_is_rejected(monkeypatch, index):
    # The other half of the pair above. 3.5e38 is the smallest "obviously over"
    # magnitude available -- close enough to the bound that a guard which went
    # looking for something dramatic like 1e39 would miss it, and far enough
    # over that no float32 rounding argument applies.
    _install(monkeypatch, lambda: _response({"embedding": _vector_with(3.5e38, index)}))
    with pytest.raises(EmbeddingBackendBadResponse, match=UNUSABLE_COMPONENT_MESSAGE):
        get_embedding(f"a response carrying 3.5e38 at index {index}")


# JSON puts no bound on an integer literal and stdlib json decodes one into a
# Python int of arbitrary size, so this is a shape a real backend can emit.
# 400 digits is far past float64's range without being so large that the
# conversion attempt itself is slow.
HUGE_INT_LITERALS = ["9" * 400, "-" + "9" * 400]


@pytest.mark.parametrize("literal", HUGE_INT_LITERALS)
@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_a_huge_integer_component_is_rejected_as_a_backend_fault(
    monkeypatch, literal, index
):
    # This case exists because of *how* it used to fail, not merely that it did.
    # math.isfinite() on an int too large to convert raises OverflowError, which
    # is an ArithmeticError -- outside _fetch()'s
    # `except (ValueError, KeyError, TypeError)` and outside every clause in
    # between. So it escaped the module entirely as a bare OverflowError and
    # surfaced as an opaque 500, breaking this module's contract that every
    # backend fault leaves as an EmbeddingBackendError. The assertion below is
    # therefore as much about the exception's *type* as about its being raised:
    # with the guard reverted this test fails by erroring out of get_embedding
    # rather than by a wrong status.
    #
    # Delivered as raw bytes because that is the production shape: stdlib json
    # has to be the thing that produces the Python int. Handing _response() a
    # ready-made Python int would test the loop while skipping the decode step
    # that is the only reason such a value can exist here at all.
    _install(monkeypatch, lambda: _raw_response(_embedding_body(literal, index)))
    with pytest.raises(
        EmbeddingBackendBadResponse, match=UNEVALUABLE_COMPONENT_MESSAGE
    ) as excinfo:
        get_embedding(f"a response carrying a {len(literal)}-digit int at {index}")
    # The one-handler contract: a caller that does not care which kind of
    # backend fault occurred catches the base class, and this must be covered
    # by that catch rather than sailing past it as an ArithmeticError.
    assert isinstance(excinfo.value, EmbeddingBackendError)


@pytest.mark.parametrize("literal", HUGE_INT_LITERALS)
@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_the_raw_payload_really_decodes_to_an_unconvertible_int(literal, index):
    """The premise the test above rests on, pinned so it cannot pass vacuously.

    Same hazard as the non-finite premise test: _fetch() translates a decode
    failure into the same exception class _validated() raises here, so a
    malformed body would make the rejection test green with the component loop
    never entered. This asserts the body parses, that it parses to the right
    width, that the component really arrives as a Python `int` (not a float, and
    not a string), and that math.isfinite() on it really raises OverflowError --
    which is the whole reason the try/except in _validated() exists.
    """
    vector = _raw_response(_embedding_body(literal, index)).json()["embedding"]
    assert len(vector) == settings.embedding_dimension
    assert type(vector[index]) is int
    with pytest.raises(OverflowError):
        math.isfinite(vector[index])


def test_a_vector_of_plausible_magnitudes_delivered_as_raw_bytes_still_passes(
    monkeypatch,
):
    # The control for the new rejections, mirroring the finite-vector control
    # above. A bound applied with the comparison inverted, or a try/except that
    # swallowed too much, would reject ordinary vectors; without a passing case
    # alongside the failing ones, every rejection test here would still be
    # green. 1e38 is a real float32 value an order of magnitude under the bound.
    last = settings.embedding_dimension - 1
    expected = _vector_with(1e38, last)
    _install(monkeypatch, lambda: _raw_response(_embedding_body("1e38", last)))
    assert get_embedding("anything") == expected
