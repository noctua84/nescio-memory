"""Unit tests for batched embedding, query embedding and error classification.

No database and no network: httpx.Client is replaced with a recording stub in
the style of tests/test_embeddings.py, so every request the module would make
is captured -- its timeout, URL and JSON payload -- and answered locally. That
is also what keeps OLLAMA_URL irrelevant here: nothing is ever dialled.
"""

import sys
import types

import httpx
import pytest

from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.embeddings import (
    OLLAMA_TIMEOUT_SECONDS,
    _fetch,
    get_embedding,
    get_embeddings,
    get_query_embedding,
)
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)

# Caller data that must never surface in an exception message: these messages
# are logged, and the texts are note content or search queries.
SECRET = "c7e2d0-caller-supplied-text-that-must-not-be-echoed"


class _RecordingClient:
    """Stands in for httpx.Client and records every construction and post.

    `respond(payload)` decides what each post returns, so a test can answer
    per request -- by echoing the inputs, failing one sub-batch, and so on.
    """

    def __init__(self, respond):
        self._respond = respond
        self.timeouts = []
        self.posts = []

    def __call__(self, *args, **kwargs):
        # The timeout is captured from the constructor because that is where
        # the module applies it; a timeout passed to post() would not count.
        self.timeouts.append(kwargs.get("timeout"))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None, **kwargs):
        self.posts.append((url, json))
        return self._respond(json)


def _install(monkeypatch, respond) -> _RecordingClient:
    client = _RecordingClient(respond)
    monkeypatch.setattr(embeddings_module.httpx, "Client", client)
    return client


@pytest.fixture(autouse=True)
def _ollama_backend(monkeypatch):
    # Pinned rather than inherited from the environment, so a developer's
    # EMBEDDING_BACKEND=local or EMBED_BATCH_SIZE cannot change what these
    # tests exercise. The local tests switch the backend themselves.
    monkeypatch.setattr(settings, "embedding_backend", "ollama")
    monkeypatch.setattr(settings, "embed_batch_size", 16)


def _vector(n: int) -> list[float]:
    """A valid full-width vector that is distinguishable by `n`."""
    vector = [0.01] * settings.embedding_dimension
    vector[0] = float(n + 1)
    return vector


def _response(payload, status_code=200):
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "http://ollama.invalid/api/embed"),
    )


def _echo(payload):
    """Answer each input text `t<n>` with _vector(n), in request order."""
    return _response(
        {"embeddings": [_vector(int(text[1:])) for text in payload["input"]]}
    )


def _constant(payload):
    """Answer every input with _vector(0), for tests that send arbitrary text."""
    return _response({"embeddings": [_vector(0)] * len(payload["input"])})


# --- sub-batching ---


def test_forty_texts_are_sent_as_sixteen_sixteen_eight_in_order(monkeypatch):
    texts = [f"t{n}" for n in range(40)]
    client = _install(monkeypatch, _echo)

    result = get_embeddings(texts)

    # One request per sub-batch, each carrying its slice in the original order.
    sent = [payload["input"] for _, payload in client.posts]
    assert [len(batch) for batch in sent] == [16, 16, 8]
    assert sent == [texts[0:16], texts[16:32], texts[32:40]]
    # And the vectors come back aligned with the texts, not merely counted.
    assert result == [_vector(n) for n in range(40)]


def test_the_payload_uses_the_batch_endpoint_shape(monkeypatch):
    client = _install(monkeypatch, _echo)

    get_embeddings(["t0", "t1"])

    # `input` as a list and an explicit truncate: with truncate false an
    # over-long passage fails as a 400, which would read as misconfiguration.
    assert client.posts == [
        (
            settings.ollama_url,
            {"model": settings.ollama_model, "input": ["t0", "t1"], "truncate": True},
        )
    ]


def test_empty_input_returns_nothing_and_makes_no_call(monkeypatch):
    client = _install(monkeypatch, _echo)

    assert get_embeddings([]) == []
    # Not even a client: an empty batch must not cost a round trip.
    assert client.timeouts == []
    assert client.posts == []


@pytest.mark.parametrize("returned", [15, 17, 0])
def test_a_count_mismatch_raises_bad_response_with_counts_only(monkeypatch, returned):
    _install(
        monkeypatch, lambda payload: _response({"embeddings": [_vector(0)] * returned})
    )

    with pytest.raises(EmbeddingBackendBadResponse) as excinfo:
        get_embeddings([SECRET] * 16)
    # Exact class: a misaligned batch is a bad answer, not a deployment fault.
    assert type(excinfo.value) is EmbeddingBackendBadResponse
    message = str(excinfo.value)
    assert str(returned) in message
    assert "16" in message
    assert SECRET not in message


def test_a_bad_vector_in_a_later_sub_batch_raises_and_returns_nothing(monkeypatch):
    def second_batch_is_wrong_width(payload):
        if payload["input"][0] == "t16":
            # One component too many: valid JSON, unusable embedding.
            return _response(
                {
                    "embeddings": [[0.01] * (settings.embedding_dimension + 1)]
                    * len(payload["input"])
                }
            )
        return _echo(payload)

    client = _install(monkeypatch, second_batch_is_wrong_width)

    # Raising is the whole assertion about "nothing returned": the first
    # sub-batch's good vectors must not leak out as a partial result.
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embeddings([f"t{n}" for n in range(40)])
    # Guards against a vacuous pass: the failure really came from the second
    # sub-batch, after the first had succeeded.
    assert [payload["input"][0] for _, payload in client.posts][:2] == ["t0", "t16"]


# --- single-text entry points ---


def test_get_embedding_equals_the_first_of_a_one_element_batch(monkeypatch):
    client = _install(monkeypatch, _echo)

    single = get_embedding("t5")
    batched = get_embeddings(["t5"])

    assert single == batched[0] == _vector(5)
    assert [payload["input"] for _, payload in client.posts] == [["t5"], ["t5"]]


def test_fetch_still_returns_a_single_vector(monkeypatch):
    # The probe measures what _fetch returns; a one-element list instead of
    # the vector would report a width of 1.
    client = _install(monkeypatch, _echo)

    assert _fetch("t3") == _vector(3)
    assert [payload["input"] for _, payload in client.posts] == [["t3"]]


# --- query embedding ---


def test_the_query_is_sent_with_the_instruction_prefix(monkeypatch):
    monkeypatch.setattr(settings, "query_instruction", "Find notes about X")
    client = _install(monkeypatch, _constant)

    assert get_query_embedding("what is pgvector") == _vector(0)

    # The exact Qwen3-Embedding format: a newline after the instruction and
    # no space after "Query:".
    assert [payload["input"] for _, payload in client.posts] == [
        ["Instruct: Find notes about X\nQuery:what is pgvector"]
    ]


def test_an_empty_instruction_sends_the_bare_query(monkeypatch):
    monkeypatch.setattr(settings, "query_instruction", "")
    client = _install(monkeypatch, _constant)

    get_query_embedding("what is pgvector")

    assert [payload["input"] for _, payload in client.posts] == [["what is pgvector"]]


def test_passages_are_never_prefixed(monkeypatch):
    # The prefix is query-side only; it must not leak into ingest embeddings,
    # which would silently invalidate every stored vector.
    monkeypatch.setattr(settings, "query_instruction", "Find notes about X")
    client = _install(monkeypatch, _echo)

    get_embeddings(["t0", "t1"])

    assert [payload["input"] for _, payload in client.posts] == [["t0", "t1"]]


def test_the_query_embedding_uses_the_short_timeout(monkeypatch):
    # 2.5 rather than the default 5, so the assertion cannot coincide.
    monkeypatch.setattr(settings, "query_embed_timeout_s", 2.5)
    client = _install(monkeypatch, _constant)

    get_query_embedding("anything")

    assert client.timeouts == [2.5]


def test_ingest_embeddings_use_the_thirty_second_timeout(monkeypatch):
    monkeypatch.setattr(settings, "query_embed_timeout_s", 2.5)
    client = _install(monkeypatch, _echo)

    get_embeddings([f"t{n}" for n in range(20)])

    # Every sub-batch, not only the first, gets the long timeout.
    assert OLLAMA_TIMEOUT_SECONDS == 30
    assert client.timeouts == [30, 30]


# --- transport failures ---
# HTTP status classification lives in tests/test_embeddings_status.py.


@pytest.mark.parametrize(
    "exception",
    [httpx.ConnectError("connection refused"), httpx.ReadTimeout("too slow")],
    ids=["connect-error", "read-timeout"],
)
def test_a_transport_failure_stays_on_the_base_class(monkeypatch, exception):
    def fail(payload):
        raise exception

    _install(monkeypatch, fail)

    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embeddings(["anything"])
    assert type(excinfo.value) is EmbeddingBackendError


# --- the local backend ---


class _Row:
    """Stands in for a numpy row: the module calls .tolist() on each."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return [_Row(_vector(int(text[1:]))) for text in texts]


def test_the_local_backend_encodes_a_list_per_sub_batch(monkeypatch):
    model = _RecordingModel()
    monkeypatch.setattr(settings, "embedding_backend", "local")
    monkeypatch.setattr(settings, "embed_batch_size", 2)
    monkeypatch.setattr(embeddings_module, "_get_local_model", lambda: model)

    result = get_embeddings(["t0", "t1", "t2"])

    # A list per call, not one string at a time: batching is the point.
    assert model.calls == [["t0", "t1"], ["t2"]]
    assert all(isinstance(call, list) for call in model.calls)
    assert result == [_vector(0), _vector(1), _vector(2)]


def _fake_sentence_transformers(monkeypatch, model_class):
    """Install a fake sentence_transformers module and clear the model cache.

    The module imports SentenceTransformer lazily inside _get_local_model, so
    a sys.modules entry is what it picks up. The cache is reset so a model
    loaded by an earlier test cannot skip the construction under test.
    """
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = model_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(embeddings_module, "_local_model", None)
    monkeypatch.setattr(settings, "embedding_backend", "local")


def test_a_local_model_that_cannot_load_is_a_misconfiguration(monkeypatch):
    class _Unloadable:
        def __init__(self, name):
            raise OSError("repo not found")

    _fake_sentence_transformers(monkeypatch, _Unloadable)

    with pytest.raises(EmbeddingBackendMisconfigured) as excinfo:
        get_embedding("anything")
    # Exact type: a typo'd model name is never fixed by retrying, so it must
    # not degrade search to FTS-only for ever as a transient fault would.
    assert type(excinfo.value) is EmbeddingBackendMisconfigured
    assert "LOCAL_EMBEDDING_MODEL" in str(excinfo.value)


def test_a_loaded_local_model_failing_to_encode_stays_transient(monkeypatch):
    # The other half of the local rule, through the real _get_local_model:
    # construction succeeds, encode() fails at runtime (memory pressure, say).
    class _OutOfMemory:
        def __init__(self, name):
            pass

        def encode(self, texts):
            raise RuntimeError("CUDA out of memory")

    _fake_sentence_transformers(monkeypatch, _OutOfMemory)

    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    assert type(excinfo.value) is EmbeddingBackendError
