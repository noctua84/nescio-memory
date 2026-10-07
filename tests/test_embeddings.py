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
from app.core.embeddings import (
    FLOAT32_MAX,
    _component_fault,
    _validated,
    get_embedding,
    probe_embedding,
    probe_embedding_dimension,
)
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


def _embedding_body_of_width(literal: str, width: int, index: int = 0) -> bytes:
    """`_embedding_body()` with the width named rather than taken from settings.

    Added for the probe tests, which are the only ones here that *want* a width
    the dimension check would reject: the probe's whole purpose is to measure a
    vector `_validated()` would refuse, so a wrong-width body is a legitimate
    input to it rather than a case that stops testing what it is named for.
    """
    components = ["0.01"] * width
    components[index] = literal
    return b'{"embedding": [' + ", ".join(components).encode("ascii") + b"]}"


def _embedding_body(literal: str, index: int = 0) -> bytes:
    """An Ollama-shaped body with `literal` written at `index`, hand-rolled.

    Sized from settings for the same reason the full-length cases below are: at
    any other length the dimension check fires first and the case stops testing
    what it is named for.
    """
    return _embedding_body_of_width(literal, settings.embedding_dimension, index)


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


# --- the probe: measure the backend, report faults, raise for almost nothing ---

# A width that is unmistakably not the configured one, so a probe that quietly
# returned settings.embedding_dimension instead of what it measured would fail
# rather than coincide. Derived from the setting rather than written as a
# literal so it stays wrong for any configured width.
OFF_WIDTH = settings.embedding_dimension + 7

# Every unusable component that needs hand-written wire bytes to exist: the
# bare non-standard literals, and an integer too large for float conversion
# (which stdlib json has to be the thing that decodes, as the rejection tests
# above explain).
UNUSABLE_RAW_LITERALS = NON_FINITE_LITERALS + HUGE_INT_LITERALS


def _raise_from(exception):
    """A stand-in for `_fetch` that raises, for the cases with no response."""

    def _raiser(text):
        raise exception

    return _raiser


@pytest.mark.parametrize("literal", UNUSABLE_RAW_LITERALS)
@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_the_probe_reports_the_width_of_a_vector_it_could_never_store(
    monkeypatch, literal, index
):
    """The probe's load-bearing property: it measures, it does not reject.

    Every case here is one `get_embedding()` raises on, and that asymmetry is
    the entire point. While the dimension probe ran the same validation a
    request does, a backend emitting 1024 NaNs made the operator script raise
    instead of report; with the opposite arrangement -- measuring without
    checking at all -- the script printed `OK` and exited 0 for a backend that
    could not embed anything. Reporting the width *and* the faults is the only
    answer that is useful in both directions.

    Both the width and probe_embedding_dimension()'s own return value are
    asserted, not merely the absence of a raise: a wrapper that swallowed the
    exception and returned 0, or settings.embedding_dimension, would satisfy a
    bare "does not raise" check while telling the operator nothing true.
    """
    _install(monkeypatch, lambda: _raw_response(_embedding_body(literal, index)))

    probe = probe_embedding()
    assert probe.dimensions == settings.embedding_dimension
    assert [fault.index for fault in probe.unusable] == [index]
    assert probe_embedding_dimension() == settings.embedding_dimension


@pytest.mark.parametrize("value", FLOAT32_OVERFLOW_VALUES + [3.5e38])
@pytest.mark.parametrize("index", NON_FINITE_POSITIONS)
def test_the_probe_reports_the_width_despite_an_out_of_range_component(
    monkeypatch, value, index
):
    # The same property for the values that survive httpx's encoder, driven
    # through the ordinary `_response()`/`json=` path for the reason the
    # rejection tests give: these need no non-standard JSON at all, which makes
    # them the likelier shape to arrive from a real backend.
    _install(monkeypatch, lambda: _response({"embedding": _vector_with(value, index)}))

    probe = probe_embedding()
    assert probe.dimensions == settings.embedding_dimension
    assert [fault.index for fault in probe.unusable] == [index]
    assert probe_embedding_dimension() == settings.embedding_dimension


def test_the_probe_reports_a_wrong_width_even_when_the_values_are_unusable(
    monkeypatch,
):
    """The compound failure, which is the one an operator most needs reported.

    A backend emitting garbage is usually the *wrong model*, so it is routinely
    also the wrong width. If the probe raised on the first bad component it
    would withhold the width in exactly the case where both facts matter, and
    the operator script could not tell "fix the model" apart from "fix the
    setting". The script's branch ordering rests on having both numbers at once;
    this pins that they are both available.
    """
    _install(
        monkeypatch, lambda: _raw_response(_embedding_body_of_width("NaN", OFF_WIDTH))
    )

    probe = probe_embedding()
    assert probe.dimensions == OFF_WIDTH
    # Stated explicitly as well as by value: if OFF_WIDTH ever coincided with
    # the configured width this test would be asserting nothing.
    assert probe.dimensions != settings.embedding_dimension
    assert [fault.index for fault in probe.unusable] == [0]
    assert probe_embedding_dimension() == OFF_WIDTH

    # get_embedding, by contrast, must still refuse this outright -- the
    # probe's leniency is the probe's alone and must not have leaked into the
    # request path, which has nothing to gain from a vector it cannot store.
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embedding("anything")


def test_the_probe_collects_every_fault_rather_than_stopping_at_the_first(
    monkeypatch,
):
    # _validated() stops at the first fault because the request is already
    # lost. The probe must not: "1 of 1024 components is unusable" and "1024 of
    # 1024 are" are different diagnoses -- the first suggests a quantisation
    # bug, the second a backend that is not an embedding model at all -- and
    # only the count distinguishes them.
    width = settings.embedding_dimension
    body = b'{"embedding": [' + b", ".join([b"NaN"] * width) + b"]}"
    _install(monkeypatch, lambda: _raw_response(body))

    probe = probe_embedding()
    assert probe.dimensions == width
    assert len(probe.unusable) == width
    # In order and once each, so a caller printing the first few is printing
    # the first few rather than an arbitrary sample.
    assert [fault.index for fault in probe.unusable] == list(range(width))


def test_the_probe_finds_a_fault_at_the_last_index_and_names_it_correctly(
    monkeypatch,
):
    """Guards two mistakes at once: a loop that stops early, and an off-by-one.

    A probe that examined only the leading components would still pass every
    index-0 case above while missing the realistic one, where a degenerate
    component sits at the end of a 1024-wide vector. And `index` is asserted by
    value because the script prints it: an off-by-one would send an operator to
    inspect the wrong component of the backend's own output, which is worse
    than printing no index at all.
    """
    last = settings.embedding_dimension - 1
    _install(monkeypatch, lambda: _raw_response(_embedding_body("NaN", last)))

    probe = probe_embedding()
    assert probe.dimensions == settings.embedding_dimension
    assert len(probe.unusable) == 1
    assert probe.unusable[0].index == last
    assert UNUSABLE_COMPONENT_MESSAGE in probe.unusable[0].message


def test_the_probe_reports_no_faults_for_a_vector_the_service_would_accept(
    monkeypatch,
):
    # The control. Every other probe test here asserts that something was
    # found, so a _component_fault() that objected to everything -- an inverted
    # comparison, say -- would leave them all green while the script reported
    # UNUSABLE VALUES for a healthy backend and exited 3 forever.
    vector = [0.01] * settings.embedding_dimension
    _install(monkeypatch, lambda: _response({"embedding": vector}))

    probe = probe_embedding()
    assert probe.dimensions == settings.embedding_dimension
    assert probe.unusable == ()
    assert probe_embedding_dimension() == settings.embedding_dimension


@pytest.mark.parametrize(
    "not_a_list, type_name",
    [("notavector", "str"), (None, "NoneType"), ({"a": 1}, "dict"), (7, "int")],
)
def test_the_probe_still_raises_when_there_is_no_vector_to_measure(
    monkeypatch, not_a_list, type_name
):
    # The one fault the probe cannot report around, and therefore the one case
    # where raising is still right: with no list there is no width to measure
    # and no components to inspect, so there is nothing to hand back. The
    # message is matched on so this cannot pass by way of some other failure --
    # _fetch() raises the same class for a body that never parsed.
    _install(monkeypatch, lambda: _response({"embedding": not_a_list}))

    with pytest.raises(
        EmbeddingBackendBadResponse,
        match=f"embedding was {type_name}, expected a list",
    ):
        probe_embedding()
    with pytest.raises(EmbeddingBackendBadResponse):
        probe_embedding_dimension()


@pytest.mark.parametrize(
    "exception",
    [
        EmbeddingBackendError("ollama refused the connection"),
        EmbeddingBackendMisconfigured("local-embeddings extra not installed"),
    ],
)
def test_the_probe_lets_a_backend_failure_through_unchanged(monkeypatch, exception):
    # The probe reports unusable *values*; it has nothing to report when there
    # are no values at all, and the operator script keys its "could not check"
    # exit code off exactly these classes. Reclassifying one here -- wrapping a
    # misconfiguration as its transient parent, say -- would make the script's
    # own diagnostic line lie about which of the two occurred.
    monkeypatch.setattr(embeddings_module, "_fetch", _raise_from(exception))

    with pytest.raises(EmbeddingBackendError) as excinfo:
        probe_embedding()
    # Exact type, not isinstance: pytest.raises matches subclasses, so the
    # clause above alone would not catch a probe that reclassified a
    # misconfiguration as its transient parent.
    assert type(excinfo.value) is type(exception)


def test_the_probe_asks_the_backend_for_the_fixed_probe_text(monkeypatch):
    # The measurement is only comparable between runs if the input is constant,
    # which is the whole reason PROBE_TEXT is a module constant rather than a
    # parameter. Captured off the real call rather than asserted about the
    # constant itself, which would be a tautology.
    asked = []
    monkeypatch.setattr(
        embeddings_module,
        "_fetch",
        lambda text: asked.append(text) or [0.01] * settings.embedding_dimension,
    )

    probe_embedding()
    assert asked == [embeddings_module.PROBE_TEXT]


# --- _component_fault as the single home of the float4 rule ---

# Values spanning every branch of the rule, each labelled rather than left to
# repr(): the huge-int cases are 400 characters wide and would otherwise make
# the parametrisation ids unreadable.
#
# Deliberately *not* annotated with the expected verdict. The point of this
# parametrisation is not to restate the rule a third time -- the rejection and
# acceptance tests above already pin it from the request side -- but to assert
# that _component_fault() and _validated() cannot disagree about any of them.
# An expected-verdict column would turn it back into a restatement, and would
# have to be edited in lockstep with the very drift it exists to catch.
AGREEMENT_VALUES = [
    pytest.param(0.0, id="zero"),
    pytest.param(0.01, id="ordinary"),
    pytest.param(-0.01, id="ordinary-negative"),
    pytest.param(1e38, id="large-but-storable"),
    pytest.param(FLOAT32_MAX, id="float32-max"),
    pytest.param(-FLOAT32_MAX, id="float32-min"),
    pytest.param(0, id="int-zero"),
    pytest.param(-1, id="int-negative"),
    pytest.param(3.5e38, id="just-over-float32-max"),
    pytest.param(-3.5e38, id="just-under-float32-min"),
    pytest.param(1e39, id="float32-overflow"),
    pytest.param(1e300, id="float32-overflow-absurd"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="negative-inf"),
    pytest.param(int("9" * 400), id="huge-int"),
    pytest.param(-int("9" * 400), id="huge-negative-int"),
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param(None, id="none"),
    pytest.param("0.01", id="numeric-string"),
    pytest.param([0.01], id="nested-list"),
    pytest.param({"a": 1}, id="object"),
]


@pytest.mark.parametrize("value", AGREEMENT_VALUES)
def test_component_fault_and_validated_never_disagree_about_a_value(value):
    """The refactor's actual guarantee: one rule, two callers, no drift.

    _validated() rejects a request and probe_embedding() diagnoses a backend,
    and while each had its own copy of the predicate a probe could measure 1024
    NaNs as a healthy 1024-wide backend while every request against it failed.
    A second copy would drift again, and it would drift silently: both halves
    would keep passing their own tests.

    So this asserts the agreement rather than the verdict. Every value
    _component_fault() objects to is one _validated() refuses, every value it
    passes is one _validated() accepts, and -- because the message is the
    sentence an operator reads -- the two describe the fault with the same
    words. FLOAT32_MAX comes from the module for the reason the boundary test
    above gives: a re-typed literal would start failing like an implementation
    bug when the bound is legitimately corrected.
    """
    fault = _component_fault(0, value)
    vector = _vector_with(value)

    if fault is None:
        assert _validated(vector) == vector
        return

    with pytest.raises(EmbeddingBackendBadResponse) as excinfo:
        _validated(vector)
    # Same words, not merely the same verdict: UnusableComponent.message exists
    # so the API and the script cannot describe one fault two ways.
    assert str(excinfo.value) == fault.message
    if fault.cause is not None:
        # The OverflowError chain, which is the only thing that explains *why*
        # a 400-digit integer could not even be evaluated. _validated() has to
        # re-raise `from` it or the traceback stops at this module.
        assert isinstance(excinfo.value.__cause__, type(fault.cause))


def test_the_agreement_cases_cover_both_verdicts():
    # Without this the test above could pass vacuously: a _component_fault()
    # that returned None for everything, or a fault for everything, would still
    # "agree" with a _validated() broken in the same direction, and nothing in
    # the parametrisation itself says both branches are exercised.
    #
    # `.values[0]` unwraps pytest.param, which is how the cases above carry
    # their readable ids.
    verdicts = {
        _component_fault(0, case.values[0]) is None for case in AGREEMENT_VALUES
    }
    assert verdicts == {True, False}
