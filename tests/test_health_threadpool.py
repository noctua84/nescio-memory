"""GET /health must answer while every threadpool token is taken.

The outage this guards: /health is the liveness probe, and a sync route runs in the
same FIFO anyio threadpool (40 tokens) as all API work. Under a burst of searches it
queued behind them; measured on real uvicorn it took 1.6 to 10 s, past the 1 to 3 s
timeout of a typical orchestrator probe, so a healthy replica could be restarted
exactly when it was busiest. The route does no blocking work, so as `async def` it
takes no token at all.

This is in-process on purpose. Here the saturation is exact: the test holds every
token itself, so a sync /health can never complete, whatever the machine's timing.
Reproducing the same thing with real load is a race that can pass by luck.
tests/test_burst_stall.py is the real-server counterpart for the pool stall.

Why the numbers are what they are:

- PROBE_TIMEOUT_S: inside the 1 to 3 s range of real probe timeouts. A healthy
  /health answers in milliseconds, so this is only ever waited out when the route
  has gone back to needing a token.
- SATURATION_TIMEOUT_S: how long the blockers get to take every token. They need
  milliseconds; it is bounded so a broken setup fails rather than hangs.

Everything runs inside one asyncio.run, because anyio keeps one default limiter per
event loop: the blockers and the app have to share the loop to share the tokens.
"""
import asyncio
import threading

import anyio.to_thread
import httpx
import pytest

from app.main import create_app

PROBE_TIMEOUT_S = 2.0
SATURATION_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.01


async def _probe_with_pool_saturated() -> httpx.Response | None:
    """GET /health with every token held. None means the probe timed out."""
    limiter = anyio.to_thread.current_default_thread_limiter()
    release = threading.Event()
    blockers = [
        asyncio.create_task(anyio.to_thread.run_sync(release.wait))
        for _ in range(int(limiter.total_tokens))
    ]
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SATURATION_TIMEOUT_S
        while limiter.available_tokens > 0 and loop.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL_S)
        # Without this the test could pass on a pool that was never saturated.
        assert limiter.available_tokens == 0, "blockers did not take every token"

        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            try:
                return await asyncio.wait_for(client.get("/health"), timeout=PROBE_TIMEOUT_S)
            except TimeoutError:
                return None
    finally:
        # Always, or a failure would leave threads parked and hang the suite at exit.
        release.set()
        await asyncio.gather(*blockers)


def test_health_answers_while_the_threadpool_is_saturated():
    response = asyncio.run(_probe_with_pool_saturated())

    if response is None:
        pytest.fail(
            f"GET /health did not answer within {PROBE_TIMEOUT_S}s while every threadpool "
            "token was held: the route is queuing for a token, so a liveness probe would "
            "time out under load"
        )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
