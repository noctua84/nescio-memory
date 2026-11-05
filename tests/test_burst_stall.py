"""A burst of searches against a real server must not stall the connection pool.

The outage this guards: a request parked between two threadpool tasks held a pooled
connection while holding no thread, those holders queued for a thread behind requests
whose threads were blocked waiting for a holder's connection, and only the 30 s
pool_timeout broke the cycle (see get_db in app/core/db.py). It needs real threads, a
real pool and a real event loop, so nothing in-process reproduces it: this starts
uvicorn in a subprocess, on the test container, and fires real concurrent HTTP at it.
tests/test_connection_release.py pins the invariant itself, without load.

Why the numbers are what they are:

- BURST_SIZE: far past both anyio's 40 tokens and the default pool's 15 connections
  (5 + 10 overflow), which is what lets holders accumulate. On the unfixed code 200
  stalled in every run measured, for 120 s and more; a healthy burst takes about 3 s.
- CLIENT_TIMEOUT_S: below the 30 s pool_timeout, so that a stall reads as a failed
  request rather than as a very slow answer that might pass.
- MID_BURST_AFTER: the other tenant's search waits until the server is demonstrably
  busy. MID_BURST_BOUND_S sends it regardless, since whether a stall reaches other
  tenants is exactly what it measures.
"""
import asyncio
import os
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models.api_key import ApiKey
from app.models.ingest_event import IngestEvent
from app.models.note import Note
from app.models.passage import Passage
from app.models.rate_limit import RateLimitWindow
from tests.factories import make_api_key, make_note, make_passage

BURST_SIZE = 200
CLIENT_TIMEOUT_S = 25.0
MID_BURST_AFTER = 20
MID_BURST_BOUND_S = 5.0

# Only ever waited out when something is already wrong.
STARTUP_TIMEOUT_S = 60.0
SHUTDOWN_TIMEOUT_S = 15.0

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERY = "pelicanium"
NOTE_COUNT = 3


# --- The server ---------------------------------------------------------------


@dataclass
class Server:
    base_url: str
    log_path: Path

    def log_tail(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _kill(process: subprocess.Popen) -> None:
    """Kill the server, whichever process it really is.

    On Windows a venv's python.exe is a launcher that runs the real interpreter as a
    child. Today's uv launcher happens to take its child down with it (measured), which
    is its own implementation detail, so /T asks for the tree outright, and the caller
    checks the port rather than trusting either. On Linux it is the interpreter itself.
    """
    if os.name == "nt":
        # check=False: it reports failure for a tree that has already exited, and the
        # port check in the caller is what decides whether anything survived.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            timeout=SHUTDOWN_TIMEOUT_S,
            check=False,
        )
    process.kill()
    process.wait(timeout=SHUTDOWN_TIMEOUT_S)


def _assert_stopped_listening(port: int) -> None:
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return
        assert time.monotonic() < deadline, (
            f"the server is still listening on port {port} after being killed: it has "
            "been orphaned, and would hold connections to the shared test container"
        )
        time.sleep(0.1)


def _wait_until_healthy(process: subprocess.Popen, server: Server) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            pytest.fail(f"the server exited with {code} during startup:\n{server.log_tail()}")
        try:
            health = httpx.get(f"{server.base_url}/health", timeout=2.0, trust_env=False)
            if health.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    pytest.fail(f"no healthy server within {STARTUP_TIMEOUT_S} s:\n{server.log_tail()}")


@pytest.fixture(scope="module")
def server(pg_container, engine, tenants, tmp_path_factory):
    """One real uvicorn for both tests.

    Depends on `tenants` so that it is torn down first: a server that stalled may still
    be working through requests, and must be gone before their rows are deleted, or it
    could write some back into the container the other tests share.
    """
    port = _free_port()
    log_path = tmp_path_factory.mktemp("burst_stall") / "server.log"
    built = Server(f"http://127.0.0.1:{port}", log_path)
    env = {
        **os.environ,
        "DATABASE_URL": pg_container.get_connection_url(),
        # The full-text arm alone, so no Ollama is needed. FTS_WEIGHT is stated because
        # the one thing the config refuses is both weights at 0, and .env could say so.
        "VECTOR_WEIGHT": "0",
        "FTS_WEIGHT": "1",
        "LANGFUSE_TRACING_ENABLED": "false",
        # This is about the pool: the default limit would refuse most of the burst.
        "RATE_LIMIT_SEARCH_REQUESTS": "10000",
        "LOG_LEVEL": "WARNING",
        "PYTHONUNBUFFERED": "1",
    }
    # A file, not a pipe: nobody reads a pipe during the burst, and a server blocked on
    # a full one would look exactly like the stall under test.
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_until_healthy(process, built)
            yield built
        finally:
            _kill(process)
            _assert_stopped_listening(port)


# --- The tenants --------------------------------------------------------------


@dataclass
class Tenants:
    noisy_name: str
    noisy_key: str
    other_key: str


@pytest.fixture(scope="module")
def tenants(engine):
    """Two committed clients; only the noisy one has notes for the burst to find.

    Committed because a separate process reads them, and so deleted by client_name
    afterwards, because the container is shared with the rest of the suite.
    """
    suffix = uuid.uuid4().hex
    noisy, other = f"burst-noisy-{suffix}", f"burst-other-{suffix}"
    with Session(bind=engine) as db:
        noisy_key = make_api_key(db, noisy, f"nm_{noisy}")
        other_key = make_api_key(db, other, f"nm_{other}")
        for number in range(NOTE_COUNT):
            note = make_note(db, noisy, path_or_uri=f"docs/{number}.md")
            make_passage(db, note, content=f"The {QUERY} isotope decays, part {number}.")
        db.commit()
    try:
        yield Tenants(noisy, noisy_key, other_key)
    finally:
        with engine.begin() as connection:
            for model in (IngestEvent, Passage, Note, RateLimitWindow, ApiKey):
                connection.execute(delete(model).where(model.client_name.in_((noisy, other))))


def _set_last_used(engine, client_name: str, value: datetime | None) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(ApiKey).where(ApiKey.client_name == client_name).values(last_used_at=value)
        )


# --- The burst ----------------------------------------------------------------


@dataclass
class Outcome:
    # The HTTP status, or the name of the exception for a request that got none.
    status: int | str
    latency_s: float
    results: int = 0


@dataclass
class Burst:
    noisy: list[Outcome]
    other: Outcome
    finished_when_other_started: int
    elapsed_s: float

    def summary(self, server: Server) -> str:
        statuses = Counter(outcome.status for outcome in self.noisy)
        latency = max(outcome.latency_s for outcome in self.noisy)
        return (
            f"burst of {BURST_SIZE}: statuses {dict(statuses)}, max latency {latency:.1f} s, "
            f"{self.elapsed_s:.1f} s in all; other tenant: {self.other.status} in "
            f"{self.other.latency_s:.1f} s, sent after {self.finished_when_other_started} "
            f"responses\n--- server log ---\n{server.log_tail()}"
        )


async def _search(client: httpx.AsyncClient, key: str) -> Outcome:
    started = time.perf_counter()
    try:
        response = await client.post(
            "/api/v1/search", json={"q": QUERY, "top_k": 5}, headers={"X-API-Key": key}
        )
    except httpx.HTTPError as exc:
        return Outcome(type(exc).__name__, time.perf_counter() - started)
    latency = time.perf_counter() - started
    results = len(response.json()["results"]) if response.status_code == 200 else 0
    return Outcome(response.status_code, latency, results)


async def _burst(server: Server, tenants: Tenants) -> Burst:
    finished = 0
    mid_burst = asyncio.Event()
    # No keep-alive and no cap: every request is its own connection, all at once, as
    # the same number of separate clients would be. A client-side pool would queue the
    # burst here instead of in the server.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=0)

    async with httpx.AsyncClient(
        base_url=server.base_url,
        limits=limits,
        timeout=httpx.Timeout(CLIENT_TIMEOUT_S),
        trust_env=False,
    ) as client:

        async def noisy_search() -> Outcome:
            nonlocal finished
            try:
                return await _search(client, tenants.noisy_key)
            finally:
                finished += 1
                if finished >= MID_BURST_AFTER:
                    mid_burst.set()

        started = time.perf_counter()
        noisy = [asyncio.create_task(noisy_search()) for _ in range(BURST_SIZE)]
        try:
            await asyncio.wait_for(mid_burst.wait(), MID_BURST_BOUND_S)
        except TimeoutError:
            pass
        finished_when_other_started = finished
        other = await _search(client, tenants.other_key)
        noisy_outcomes = await asyncio.gather(*noisy)
        elapsed = time.perf_counter() - started

    return Burst(noisy_outcomes, other, finished_when_other_started, elapsed)


def _assert_nobody_stalled(burst: Burst, server: Server) -> None:
    summary = burst.summary(server)
    assert burst.finished_when_other_started < BURST_SIZE, f"not mid-burst: {summary}"
    assert {outcome.status for outcome in burst.noisy} == {200}, summary
    # A httpx timeout is per read, so a slow answer can outlast it and still be a 200.
    assert max(outcome.latency_s for outcome in burst.noisy) < CLIENT_TIMEOUT_S, summary
    # Proof the searches did real work, rather than all taking some shorter path.
    assert all(outcome.results == NOTE_COUNT for outcome in burst.noisy), summary

    assert burst.other.status == 200, summary
    assert burst.other.latency_s < CLIENT_TIMEOUT_S, summary


# --- The tests ----------------------------------------------------------------


def test_a_burst_from_one_client_finishes_and_leaves_other_tenants_served(
    engine, server, tenants
):
    """The ordinary case: a recently used key, so no request needs a touch."""
    _set_last_used(engine, tenants.noisy_name, datetime.now(UTC))

    burst = asyncio.run(_burst(server, tenants))

    _assert_nobody_stalled(burst, server)


def test_a_burst_on_a_key_with_no_recorded_use_does_not_nest_connections(
    engine, server, tenants
):
    """A key's first use: a touch is due for every request that authenticates before the
    first touch commits, which in a burst is everyone in the first wave. A touch nested
    in the auth transaction takes a second connection under the first, so the first wave
    can hold the whole pool with each waiting for one more, and then no touch can commit
    to end that wave at all."""
    _set_last_used(engine, tenants.noisy_name, None)

    burst = asyncio.run(_burst(server, tenants))

    _assert_nobody_stalled(burst, server)
    with engine.connect() as connection:
        last_used = connection.execute(
            select(ApiKey.last_used_at).where(ApiKey.client_name == tenants.noisy_name)
        ).scalar_one()
    assert last_used is not None, "no touch ever landed, so this tested nothing"
