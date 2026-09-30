# Dependency Failure Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `nescio-memory` defined HTTP behaviour when Ollama or PostgreSQL fail, instead of an unhandled `500`.

**Architecture:** `app/core/embeddings.py` wraps its transport failures in domain exceptions so `httpx` never reaches the HTTP layer. A new `app/core/errors.py` defines those exceptions, each carrying the status code and detail it maps to, plus a single `register_exception_handlers(app)` wired into `create_app()`. SQLAlchemy's `OperationalError` is handled directly, since SQLAlchemy is already the database abstraction.

**Tech Stack:** FastAPI exception handlers, httpx, SQLAlchemy, pytest against the existing testcontainers harness.

**Spec:** `docs/superpowers/specs/2026-09-29-stabilization-design.md` — this plan implements **item 1 only**. Items 2-7 are separate changes in the same release.

## Global Constraints

- The response body shape is **`{"detail": "..."}`** — matching the existing 400 and 401 responses. Do not add a `code` field and do not use `problem+json`.
- **No retries.** The first embedding failure aborts the request. Embedding runs once per chunk, so retries multiply against chunk count.
- `Retry-After` is a **fixed 30 seconds**, defined as one constant.
- Exact detail strings, used verbatim:
  - `Embedding backend unavailable`
  - `Embedding backend returned an unexpected response`
  - `Database unavailable`
  - `Embedding backend is misconfigured`
- The misconfigured case returns **500 and no `Retry-After`**. A missing extra will never succeed on retry, so telling the client to come back would be false.
- `httpx` must not be imported by `app/main.py` or `app/core/errors.py`. Transport detail stays inside `app/core/embeddings.py`.
- Client-facing `detail` strings come from the exception **class**, never from `str(exc)`. Exception messages may carry internal detail for logs and must not reach the client.
- `/health` is unchanged. It stays liveness-only and contacts nothing.
- Do not add a readiness probe, do not make the Ollama timeout configurable, and do not touch `MAX_CONTENT_CHARS`, the embedding-dimension check, or any index. Those are items 2-7.
- Never add `ruff`, `mypy`, or any other checker.
- Commit prefix: `fix: [fix] <subject>` for production code, `test: [test] <subject>` for test-only commits.

## File Structure

| File | Responsibility |
|---|---|
| `app/core/errors.py` | **new** — domain exception classes carrying their HTTP mapping, and `register_exception_handlers(app)` |
| `app/core/embeddings.py` | modify — wrap `httpx` and import failures in those exceptions |
| `app/main.py` | modify — call `register_exception_handlers(app)` inside `create_app()` |
| `tests/test_embeddings.py` | **new** — unit tests that `get_embedding` raises the right domain exception; no database needed |
| `tests/test_error_responses.py` | **new** — endpoint-level tests for status, detail and `Retry-After` |

Why two test files: the first exercises `get_embedding` directly with no HTTP and no container, so it runs fast and fails precisely. The second exercises the mapping through real endpoints. Conflating them would make a transport bug and a handler bug look identical.

## A note on the OpenAPI contract

This plan adds handlers but does not add `responses={...}` to any route, so `app.openapi()` output is unchanged and the committed `openapi.json` / `openapi.yaml` stay correct. The CI drift check will pass without regenerating them. Documenting the 503 in the schema would be a genuine improvement and is deliberately **not** in this plan — it is not in the spec.

---

### Task 1: Domain exceptions, and `get_embedding` raising them

**Files:**
- Create: `app/core/errors.py`
- Modify: `app/core/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all importable from `app.core.errors`:
  - `RETRY_AFTER_SECONDS: int` (value `30`)
  - `class EmbeddingBackendError(RuntimeError)` with class attributes `status_code: int`, `detail: str`, `retry_after: bool`
  - `class EmbeddingBackendBadResponse(EmbeddingBackendError)`
  - `class EmbeddingBackendMisconfigured(EmbeddingBackendError)`
  - `app.core.embeddings.get_embedding(text: str) -> list[float]` — unchanged signature, now raising the above instead of leaking `httpx` errors

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embeddings.py`:

```python
"""Unit tests for get_embedding's failure translation.

No database and no container: these call get_embedding directly with a stubbed
transport, so a failure here points at the embedding module rather than at the
HTTP layer.
"""
import httpx
import pytest

from app.core import embeddings as embeddings_module
from app.core.embeddings import get_embedding
from app.core.errors import EmbeddingBackendBadResponse, EmbeddingBackendError


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


def test_connection_failure_raises_embedding_backend_error(monkeypatch):
    def refuse():
        raise httpx.ConnectError("connection refused")

    _install(monkeypatch, refuse)
    with pytest.raises(EmbeddingBackendError):
        get_embedding("anything")


def test_timeout_raises_embedding_backend_error(monkeypatch):
    def time_out():
        raise httpx.ReadTimeout("too slow")

    _install(monkeypatch, time_out)
    with pytest.raises(EmbeddingBackendError):
        get_embedding("anything")


def test_non_2xx_raises_embedding_backend_error(monkeypatch):
    _install(monkeypatch, lambda: _response({"error": "boom"}, status_code=500))
    with pytest.raises(EmbeddingBackendError):
        get_embedding("anything")


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
    _install(monkeypatch, lambda: _response({"embedding": [0.1, 0.2, 0.3]}))
    assert get_embedding("anything") == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_embeddings.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.core.errors'`.

- [ ] **Step 3: Create `app/core/errors.py`**

```python
"""Domain exceptions and the HTTP responses they map to.

The API layer must not know how a dependency is reached. EmbeddingBackendError is
raised by app.core.embeddings whether the backend is Ollama over HTTP or
sentence-transformers in process, so one handler covers both and no transport
detail crosses a layer boundary.

Each exception carries its own mapping as class attributes, so adding a case
means adding a class rather than extending a dispatch table.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

# A dependency that is down is usually down for longer than one request, so this
# is a hint to back off rather than a promise about recovery.
RETRY_AFTER_SECONDS = 30

DATABASE_UNAVAILABLE_DETAIL = "Database unavailable"


class EmbeddingBackendError(RuntimeError):
    """The configured embedding backend could not produce a vector.

    Transient by assumption: the backend was reachable in principle and may work
    on a later attempt.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Embedding backend unavailable"
    retry_after = True


class EmbeddingBackendBadResponse(EmbeddingBackendError):
    """The backend answered, but not with an embedding we could read."""

    detail = "Embedding backend returned an unexpected response"


class EmbeddingBackendMisconfigured(EmbeddingBackendError):
    """The backend cannot work as deployed.

    Distinct from its parent because retrying will never help: the remedy is a
    deployment change, not patience. It therefore maps to 500 and sends no
    Retry-After, since advertising one would tell the client something untrue.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Embedding backend is misconfigured"
    retry_after = False


def _service_unavailable(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": detail},
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Map dependency failures onto HTTP responses. Called from create_app()."""

    @app.exception_handler(EmbeddingBackendError)
    async def _embedding_backend_failed(
        request: Request, exc: EmbeddingBackendError
    ) -> JSONResponse:
        # detail comes from the class, never from str(exc): exception messages
        # carry internal detail for logs and must not reach the client.
        headers = (
            {"Retry-After": str(RETRY_AFTER_SECONDS)} if exc.retry_after else None
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    @app.exception_handler(OperationalError)
    async def _database_unavailable(
        request: Request, exc: OperationalError
    ) -> JSONResponse:
        # SQLAlchemy is already the abstraction over the database, so handling its
        # exception directly is correct rather than leaky. OperationalError is the
        # connectivity family; programming errors are bugs and stay 500s.
        return _service_unavailable(DATABASE_UNAVAILABLE_DETAIL)
```

- [ ] **Step 4: Make `app/core/embeddings.py` raise them**

Replace the whole file with:

```python
import httpx

from app.config import settings
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)

_local_model = None

# Kept hardcoded deliberately. Because the first failure aborts the request, this
# bounds a failed ingest at roughly one timeout rather than one per chunk.
OLLAMA_TIMEOUT_SECONDS = 30.0


def _get_local_model():
    global _local_model
    if _local_model is None:
        # Imported lazily so the local-embeddings extra, and the torch it
        # pulls in, are only needed when that backend is selected.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            # The install hint is for the operator reading logs; the client sees
            # only the class's fixed detail.
            raise EmbeddingBackendMisconfigured(
                "EMBEDDING_BACKEND=local needs the local-embeddings extra; "
                "install it with `uv sync --extra local-embeddings`"
            ) from exc
        _local_model = SentenceTransformer(settings.local_embedding_model)
    return _local_model


def get_embedding(text: str) -> list[float]:
    if settings.embedding_backend == "local":
        return _get_local_model().encode(text).tolist()

    payload = {"model": settings.ollama_model, "prompt": text}
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = client.post(settings.ollama_url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        # httpx.HTTPError is the base of RequestError (connection, timeout) and
        # HTTPStatusError, so every transport failure is caught here and the
        # httpx dependency stops at this module's edge.
        raise EmbeddingBackendError(f"Ollama request failed: {exc}") from exc

    try:
        return response.json()["embedding"]
    except (ValueError, KeyError, TypeError) as exc:
        # ValueError covers a non-JSON body (JSONDecodeError subclasses it);
        # KeyError and TypeError cover JSON that is not the shape we expect.
        raise EmbeddingBackendBadResponse(
            f"Ollama response was not a usable embedding: {exc}"
        ) from exc
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_embeddings.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Confirm nothing else regressed**

Run:

```bash
uv run pytest -q
```

Expected: 37 passed (the existing 30 plus 7 new).

- [ ] **Step 7: Confirm the OpenAPI spec has not drifted**

Adding exception classes must not change the published contract.

```bash
uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml && echo "no drift"
```

Expected: `no drift`. If this reports changes, stop and report it — it means something touched a route signature.

- [ ] **Step 8: Commit**

```bash
git add app/core/errors.py app/core/embeddings.py tests/test_embeddings.py
git commit -m "fix: [fix] translate embedding backend failures into domain errors"
```

---

### Task 2: Map the domain exceptions onto HTTP responses

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_error_responses.py`

**Interfaces:**
- Consumes from Task 1: `app.core.errors.register_exception_handlers`, `EmbeddingBackendError`, `EmbeddingBackendBadResponse`, `EmbeddingBackendMisconfigured`, `RETRY_AFTER_SECONDS`.
- Produces: the HTTP contract other tasks and clients rely on — 503 with `Retry-After` for both dependencies, 500 without it for misconfiguration.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_error_responses.py`:

```python
"""Endpoint-level tests for dependency failure responses.

These go through the real routes, so they verify the wiring in create_app() as
well as the mapping itself.
"""
import pytest
from sqlalchemy.exc import OperationalError

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.core.db import get_db
from app.core.errors import (
    RETRY_AFTER_SECONDS,
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.main import app
from tests.factories import make_api_key

LONG_ENOUGH = (
    "This content comfortably exceeds the fifty character minimum that the "
    "ingest endpoint applies to each chunk."
)
INGEST_PAYLOAD = {
    "repo_name": "repo_a",
    "file_path": "docs/note.md",
    "content": LONG_ENOUGH,
}
SEARCH_PAYLOAD = {"query": "anything", "top_k": 5}


def _raise(exception):
    def _raiser(text):
        raise exception

    return _raiser


def test_ingest_returns_503_when_the_embedding_backend_is_down(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        ingest_module, "get_embedding", _raise(EmbeddingBackendError("refused"))
    )

    response = client.post(
        "/api/v1/ingest", data=INGEST_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding backend unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)

    # Nothing may be persisted. delete_by_file() runs before the first insert, so
    # a failure that left the transaction committed would lose data rather than
    # merely fail.
    from sqlalchemy import func, select

    from app.models.learning import Learning

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_search_returns_503_when_the_embedding_backend_is_down(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        search_module, "get_embedding", _raise(EmbeddingBackendError("refused"))
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding backend unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


def test_an_unreadable_backend_response_is_reported_distinctly(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        search_module,
        "get_embedding",
        _raise(EmbeddingBackendBadResponse("no embedding key")),
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Embedding backend returned an unexpected response"
    }


def test_a_misconfigured_backend_is_500_with_no_retry_after(
    client, db_session, monkeypatch
):
    # A missing extra will never succeed on retry, so advertising Retry-After
    # would tell the client something untrue.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        search_module,
        "get_embedding",
        _raise(EmbeddingBackendMisconfigured("extra not installed")),
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding backend is misconfigured"}
    assert "Retry-After" not in response.headers


def test_internal_exception_messages_do_not_reach_the_client(
    client, db_session, monkeypatch
):
    # The exception carries an internal message for logs; the response must show
    # only the class's fixed detail.
    key = make_api_key(db_session, "acme")
    secret = "http://internal-ollama.corp:11434 refused the connection"
    monkeypatch.setattr(
        search_module, "get_embedding", _raise(EmbeddingBackendError(secret))
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert secret not in response.text
    assert response.json() == {"detail": "Embedding backend unavailable"}


def test_database_failure_returns_503(db_session, monkeypatch):
    # Built without the `client` fixture, because this test needs a session that
    # fails rather than the working one that fixture installs.
    from fastapi.testclient import TestClient

    from tests.fakes import fake_embedding

    monkeypatch.setattr(search_module, "get_embedding", fake_embedding)

    class _FailingSession:
        def execute(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        def close(self):
            pass

    # The key is irrelevant: get_current_client resolves through get_db, so the
    # OperationalError is raised while authenticating, before the endpoint body.
    # A 401 here would mean the handler is not reached from a dependency.
    app.dependency_overrides[get_db] = lambda: _FailingSession()
    try:
        response = TestClient(app).post(
            "/api/v1/search",
            json=SEARCH_PAYLOAD,
            headers={"X-API-Key": "nm_irrelevant"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


def test_health_is_unaffected(client):
    # /health must keep contacting nothing, so a dependency outage cannot turn a
    # liveness probe into a restart.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_error_responses.py -v
```

Expected: the failures raise out of the endpoints rather than producing a response, because no handler is registered yet. With `TestClient`'s default `raise_server_exceptions=True` this surfaces as `EmbeddingBackendError` propagating out of `client.post`, not as an assertion failure on the status code. Confirm you see the exception, not a 500 — that is what proves nothing is handling it.

- [ ] **Step 3: Wire the handlers into `create_app()`**

In `app/main.py`, add the import and one call. The call must come **before** `include_router`, so it is obvious the handlers apply to every route including ones added later:

```python
from fastapi import FastAPI

from app.api.v1.router import api_router
from app.config import settings
from app.core.errors import register_exception_handlers
from app.helper import get_app_version

__version__ = get_app_version()

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="NescioAI semantic memory core.",
        version=__version__,
    )

    register_exception_handlers(app)

    app.include_router(api_router, prefix="/api/v1")
```

Leave the rest of the file, including `health()` and the trailing `app = create_app()`, exactly as it is.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_error_responses.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run the whole suite**

Run:

```bash
uv run pytest -q
```

Expected: 44 passed.

- [ ] **Step 6: Confirm no OpenAPI drift**

```bash
uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml && echo "no drift"
```

Expected: `no drift`. Registering handlers does not alter route signatures.

- [ ] **Step 7: Commit**

```bash
git add app/main.py tests/test_error_responses.py
git commit -m "fix: [fix] return 503 with Retry-After when a dependency is down"
```

---

### Task 3: Pin ingest atomicity on partial failure

**Corrected 2026-09-30 during execution.** The original version of this task
specified the test through the shared `client` fixture. That cannot work, and the
first attempt returned BLOCKED reporting a production data-loss bug that does not
exist. See "Why the shared fixture cannot express this" below.

**Files:**
- Modify: `tests/test_error_responses.py`

**Interfaces:**
- Consumes: the `connection` fixture from `tests/conftest.py`, `app.core.db.get_db`,
  `tests.factories.make_api_key` and `make_learning`, `app.core.errors.EmbeddingBackendError`.
- Produces: nothing other tasks depend on.

**Why this task exists:** `/api/v1/ingest` calls `repo.delete_by_file(...)` before
writing any new chunk. If an embedding fails partway through a multi-chunk file
and the transaction were committed anyway, the caller would be left with their old
chunks deleted and the new ones absent — data loss caused by a transient outage,
which is far worse than a failed request. Production is safe today because
`get_db` is a generator whose `finally: db.close()` discards the uncommitted work,
but nothing asserts it.

### Why the shared fixture cannot express this

`tests/conftest.py` overrides the dependency with `app.dependency_overrides[get_db]
= lambda: db_session` — a plain function. Production's `get_db` is a generator, and
its `finally: db.close()` is the mechanism that performs the rollback. A plain
lambda has no teardown, so in a test using the `client` fixture the session is
never closed, nothing is rolled back, and a query afterwards sees the endpoint's
uncommitted work still sitting in the session the test shares with the app.

A test written that way does not fail because production is broken. It fails
because the harness never runs the code under test. This was verified by executing
the identical scenario against an override that does close the session: the
original row survives and the failed ingest rolls back completely.

This task therefore builds its own client with a production-shaped override. Do not
"simplify" it back onto the `client` fixture.

- [ ] **Step 1: Write the test**

Add `make_learning` to the existing `from tests.factories import make_api_key` line
at the top of `tests/test_error_responses.py`, and add these imports to the same
block: `from sqlalchemy.orm import Session`, `from sqlalchemy import func, select`,
`from app.models.learning import Learning`. Then append:

```python
ORIGINAL_CONTENT = "the original content, which must survive a failed re-ingest"


def test_a_failure_partway_through_a_file_leaves_earlier_rows_intact(
    connection, monkeypatch
):
    """Ingest must be all-or-nothing.

    delete_by_file() runs before the first insert, so a failure midway could
    otherwise leave the caller's previous chunks deleted and the new ones absent.

    This test deliberately does NOT use the `client` fixture. That fixture
    overrides get_db with `lambda: db_session`, a plain function with no teardown,
    so the `finally: db.close()` that performs the rollback in production never
    runs and the rollback cannot be observed. The override below is generator
    shaped, like the real get_db.
    """
    # Setup lives in its own session and is committed, so releasing the endpoint's
    # savepoint later cannot take the setup row with it.
    setup = Session(bind=connection, join_transaction_mode="create_savepoint")
    key = make_api_key(setup, "acme")
    make_learning(
        setup,
        "acme",
        repo_name="repo_a",
        file_path="docs/note.md",
        content=ORIGINAL_CONTENT,
    )
    setup.commit()

    calls = {"n": 0}

    def fail_on_the_second_chunk(text):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise EmbeddingBackendError("backend died mid-file")
        return [0.0] * 384

    monkeypatch.setattr(ingest_module, "get_embedding", fail_on_the_second_chunk)

    def production_shaped_get_db():
        request_session = Session(
            bind=connection, join_transaction_mode="create_savepoint"
        )
        try:
            yield request_session
        finally:
            # The line under test. Without it nothing rolls back.
            request_session.close()

    app.dependency_overrides[get_db] = production_shaped_get_db
    try:
        # 2000 characters is 3 chunks at chunk_size 1000 with overlap 200, so the
        # failure lands after at least one successful embedding.
        response = TestClient(app).post(
            "/api/v1/ingest",
            data={**INGEST_PAYLOAD, "content": "x" * 2000},
            headers={"X-API-Key": key},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert calls["n"] == 2, "expected the second chunk to be the one that failed"

    check = Session(bind=connection, join_transaction_mode="create_savepoint")
    surviving = check.scalars(
        select(Learning).where(Learning.client_name == "acme")
    ).all()
    assert len(surviving) == 1
    assert surviving[0].content == ORIGINAL_CONTENT
    assert check.scalar(select(func.count()).select_from(Learning)) == 1
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_error_responses.py::test_a_failure_partway_through_a_file_leaves_earlier_rows_intact -v
```

Expected: **PASS** on the first run. The behaviour already exists; this test pins
it. If it fails, stop and report — do not adjust the assertions to match.

- [ ] **Step 3: Prove the test has teeth**

A test that has never failed has not been shown to test anything.

Temporarily delete the `finally: request_session.close()` block from the override
**inside the test**, re-run, and confirm it now fails: with no teardown, nothing
rolls back and the original row is gone. Then restore it and confirm it passes.

This mutation is entirely within the test file. Unlike the earlier draft of this
task, **no production code is touched at any point** — an unreverted edit here
cannot ship a defect.

- [ ] **Step 4: Run the whole suite**

```bash
uv run pytest -q
```

Expected: 48 passed, and `git diff app/` empty.

- [ ] **Step 5: Commit**

```bash
git add tests/test_error_responses.py
git commit -m "test: [test] pin ingest atomicity when a chunk fails mid-file"
```

---

## Verification checklist

After Task 3, confirm all of the following:

- [ ] `uv run pytest` passes — 45 tests.
- [ ] `uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml` reports no drift.
- [ ] `git status --porcelain` is empty.
- [ ] `grep -rn "httpx" app/main.py app/core/errors.py` returns nothing — transport detail did not leak.
- [ ] `grep -n "str(exc)" app/core/errors.py` returns nothing — client details come from classes, not messages.
- [ ] `/health` still contacts nothing: `grep -n "get_db\|Depends" app/main.py` shows no dependency on the health route.
- [ ] No `[tool.ruff]` or `[tool.mypy]` was added.
- [ ] Items 2-7 of the spec are untouched: no change to `MAX_CONTENT_CHARS`, no dimension check, no index, no CI edit.
