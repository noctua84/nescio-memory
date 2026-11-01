---
name: Resetting a fake clock does not drain its queued one-shot callbacks
description: Synthetic fixture. tickmock's clock.clear() drops recorded ticks but keeps callbacks registered with onceAfter(); an unconsumed one-shot leaks into the next test — use clock.reset() in setup for clocks that get one-shot registrations
type: feedback
---
<!-- promoted:begin -->
In the invented `tickmock` library, `clock.clear()` only discards the recorded
tick log. It does **not** drop callbacks registered through `onceAfter()`.

So in a suite where one case registers `clock.onceAfter(50, raise_timeout)` but
its code path never advances far enough to consume it, the one-shot survives the
setup hook that "cleans up" with `clock.clear()` and fires inside **whichever
case runs next** — a failure that surfaces in an unrelated test and that moves
when cases are reordered or run alone.

Fixes:
- Use `clock.reset()` in the setup hook for any clock a case overrides with a
  one-shot, then re-register the default callbacks.
- Or assert in the registering case that the one-shot actually fired, so an
  unconsumed registration fails where it was written.

Observed while writing this fixture; the library, the API and the date are all
invented for test purposes.
[Source: empirical — 2026-09-30]
<!-- promoted:end -->
