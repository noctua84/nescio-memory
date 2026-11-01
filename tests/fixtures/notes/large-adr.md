---
name: example-project-adr-0007-bounded-cache-eviction
description: Synthetic fixture — an invented architecture decision record for the fictional `cachekit` in-process cache library, in which eviction moves from unbounded memoisation to a three-tier bounded policy with explicit admission control. Long, many sections, fenced code, no provenance line.
type: adr
---
# ADR-0007: Bounded cache eviction for the cachekit library

- **Status:** Accepted (direction confirmed by the fictional maintainer, 2031-02-14). **IMPLEMENTED — shipped in the invented 4.0 release.** See *Amendment: tier boundaries relaxed* at the end of this document.
- **Date:** 2031-02-14
- **Deciders:** The maintainer of this fixture · nobody real
- **Scope:** The eviction policy of `cachekit.Cache`. The admission filter and the persistence sidecar are in scope; the serialisation format is not.
- **Supersedes:** the unbounded-memoisation portion of ADR-0004. Does **not** supersede ADR-0002 (key hashing), which is preserved intact.

Everything in this file is invented for the purpose of exercising the splitter
on a long, heavily sectioned document. There is no provenance marker anywhere
in it, deliberately, because one test asserts that a body without a provenance
line reports neither provenance field.

## Context

`cachekit.Cache` began life as a decorator that memoised a pure function for
the lifetime of the process. That was adequate while the keyspace was small and
every value was a few hundred bytes. It stopped being adequate once callers
started caching rendered documents keyed by request parameters, because the
keyspace became effectively unbounded and the process grew until the host
killed it.

The first attempt at a fix was a maximum entry count. It helped, and then it
stopped helping, because entry count is a poor proxy for memory when entry
sizes differ by four orders of magnitude. A cache of ten thousand small entries
and a cache of ten thousand rendered documents are the same cache by that
measure and nothing alike in practice.

The second attempt was a byte budget with least-recently-used eviction. That
removed the crashes and introduced a new complaint: a single sweep of cold keys
would evict every warm entry, because a scan touches each key exactly once and
LRU cannot tell a scan from a change in the access pattern. Throughput after a
nightly report run dropped by a factor the fictional users described, with some
justification, as insulting.

## Decision

The cache keeps a byte budget, an admission filter in front of it, and three
eviction tiers behind it. An entry is admitted only if the filter has seen its
key more than once; once admitted it starts in the probation tier and is
promoted on its second hit inside the tier.

```python
cache = Cache(
    budget_bytes=256 * 1024 * 1024,
    admission="frequency",
    tiers=("probation", "protected", "pinned"),
)
```

The budget is the only required parameter. Everything else has a default that
is correct for the common case of a few thousand medium entries, and every
default is printed by `cache.explain()` together with the layer it came from.

### The three eviction tiers

The probation tier holds newly admitted entries and is the only tier eviction
ever draws from while it is non-empty. It is sized as a fraction of the budget
rather than a fraction of the entry count, so a probation tier full of large
entries holds fewer of them, which is the intent: a large cold entry is the
most expensive thing the cache can be wrong about.

The protected tier holds entries that have been hit at least twice since
admission. Entries leave it by demotion to probation, never by direct
eviction, so a warm entry always gets one more chance to prove itself before
it is discarded. A scan therefore costs the probation tier and leaves the
protected tier intact, which is exactly the complaint that motivated this
record.

The pinned tier holds entries the caller has explicitly marked as never
evictable. It is capped at a configurable fraction of the budget and the cap is
enforced at pin time, not at eviction time, so a caller that pins too much gets
an error at the call site instead of a cache that silently stops evicting.

### Configuration surface

Three new parameters, all keyword-only, all with defaults:

```python
Cache(
    budget_bytes=...,          # required
    admission="frequency",     # or "always", or a callable
    probation_fraction=0.2,    # of budget_bytes
    pinned_fraction=0.1,       # of budget_bytes
    sizer=default_sizer,       # bytes for one value
)
```

The `sizer` hook exists because the library cannot measure the footprint of an
arbitrary object, and guessing wrong in either direction is worse than asking.
A caller that stores uniform values passes a constant and skips the
measurement entirely.

### Metrics emitted

Every decision the policy makes is counted: admissions, rejections, promotions,
demotions, evictions by tier, and pin failures. The counters are plain integers
on the cache object rather than a reporting integration, because a library that
chooses a metrics backend for its host is a library the host will eventually
have to work around.

A derived hit ratio is offered as a property, computed rather than stored, so
that it can never drift from the counters it summarises.

## Eviction algorithm

Eviction runs when an insertion would push the resident byte total past the
budget, and it runs until the total is back under it. It never runs on a timer
and never runs on a background thread, because both make the cost of an
insertion unpredictable and both make a test of eviction a test of scheduling.

The first step is to compute the overshoot: the number of bytes by which the
insertion would exceed the budget, plus a small slack so that a steady stream
of same-sized insertions does not evict on every single call. The slack is a
fraction of the budget rather than a fixed byte count, so it scales with the
cache instead of becoming irrelevant on a large one and dominant on a small
one.

The second step is to drain the probation tier in least-recently-used order
until either the overshoot is covered or the tier is empty. Draining a tier
means removing its least recently used entry, subtracting that entry's recorded
size from the resident total, and emitting an eviction count against the tier.
The entry's recorded size is used rather than a fresh measurement, because a
value that changed size after insertion would otherwise corrupt the running
total in a way no later measurement can repair.

The third step, reached only when the probation tier is empty and the
overshoot is not covered, is to demote from the protected tier into probation
and then continue draining. Demotion is deliberately a separate step from
eviction: it gives a warm entry one more interval in which a hit can save it,
and it means a single oversized insertion cannot clear the whole protected tier
in one pass.

The fourth step is the failure case. If the probation tier is empty, the
protected tier is empty, and the overshoot is still not covered, then the
pinned tier plus the incoming entry exceed the budget on their own. The
insertion fails with an error that names the budget, the pinned total and the
size of the incoming entry, in that order. It does not evict a pinned entry,
because a pin that can be broken by load is not a pin and a caller that relied
on one would have no way to find out that it had been broken.

Between the second and third steps the algorithm consults the admission filter
one further time, for the single case of an insertion that is itself larger
than the probation fraction of the budget. Such an entry cannot fit in
probation at all, so admitting it would guarantee an immediate second eviction
pass. Instead it is admitted directly into the protected tier if the filter has
seen its key at least three times, and rejected outright otherwise. This is the
one place where the policy is asymmetric, and it is asymmetric on purpose: the
expensive mistake with a very large entry is admitting it, not rejecting it.

The whole pass holds one lock for its duration. Fine-grained locking was
measured and rejected, because the pass is short, the contention is low, and
two locks taken in two orders by two tiers is the sort of defect that reproduces
once a month in production and never in a test. The lock is documented as an
implementation detail and is not exposed.

Recorded sizes are maintained incrementally. Inserting an entry adds its size,
evicting one subtracts it, and overwriting one subtracts the old size before
adding the new. A periodic consistency check, enabled only under the library's
own test suite, recomputes the total from the resident entries and raises if it
disagrees, so that an arithmetic mistake in any of those three paths is caught
by the next test run rather than by a host whose cache slowly forgets how large
it is.

## Alternatives considered

Three alternatives were considered seriously enough to be written down, each
of which is simpler than the chosen policy and each of which fails on one of
the two workloads the fixture's fictional users actually run: a steady stream
of repeated keys, and a nightly scan that touches every key exactly once.

### Alternative: pure least-recently-used

The simplest possible policy, and the one the library already had. It is
correct for the steady stream and catastrophic for the nightly scan: the scan
touches every cold key once, every touch makes that key the most recently used,
and by the end of the scan the warm working set has been evicted in its
entirety. Recovery then takes as long as the working set takes to rebuild,
which on the fictional workload is most of the following morning.

It was rejected for exactly this reason, not for its hit ratio in the steady
state, where it is as good as anything else.

### Alternative: pure least-frequently-used

Protects the working set from a scan, because a key touched once never
accumulates enough frequency to displace a key touched thousands of times. It
fails on the opposite axis: an entry that was extremely popular yesterday and
is irrelevant today keeps its accumulated count and cannot be displaced by a
newly popular entry at all.

Decaying the counts fixes this and introduces a decay interval, which is one
more parameter nobody can choose correctly without measuring their own
workload. The tiered policy achieves the same protection without asking the
caller to pick a half-life.

### Alternative: time-to-live only

Attractive because it is trivial to explain and trivial to test. Rejected
because it bounds staleness and not memory: a burst of insertions inside one
interval exceeds any budget regardless of the interval's length, and shrinking
the interval until it cannot does so by making the cache useless. Time-to-live
remains available as an orthogonal option and is not a substitute for a budget.

## Consequences

Memory is bounded and scans no longer evict the working set. The policy is
harder to explain than what it replaced.

## Migration plan

Callers constructing `Cache()` without a budget keep the old unbounded
behaviour for one release, with a deprecation warning that names the file and
line of the construction. In the following release the parameter becomes
required and the unbounded code path is deleted rather than kept behind a flag,
because an unbounded path that still exists is an unbounded path somebody will
still be running.

Callers who had set the old maximum entry count get a translation: the count is
multiplied by a measured average entry size and used as the budget, with the
computation reported once at startup so that an obviously wrong translation is
visible immediately instead of being discovered under load.

## Rollout stages

Stage one enables the budget with admission disabled, which bounds memory
without changing which entries are kept. Stage two enables the frequency
admission filter. Stage three enables the protected tier. Each stage is a
separate release and each is independently revertible by configuration, so a
regression can be attributed to one change rather than to three.

The stages are deliberately ordered so that the change most likely to cause a
complaint — admission, which can refuse to cache something the caller expected
to be cached — lands in the middle, with a known-good configuration on either
side of it.

## Interaction with the persistence sidecar

The optional sidecar writes evicted entries to disk so that a later miss can be
served from a local file instead of recomputed. Eviction therefore has a cost
beyond the loss of the entry, and the policy must not treat writing to the
sidecar as free.

The resolution is that the sidecar write is queued, not performed inline, and
that a full queue causes the write to be dropped rather than the eviction to be
delayed. An eviction that cannot complete promptly is worse than a sidecar that
is occasionally incomplete, because the former blocks an insertion the caller
is waiting on.

## Thread-safety notes

The cache is safe for concurrent use by threads and makes no claim about
processes. Every mutation takes the single lock described above; every read
takes it too, because the recency bookkeeping a read performs is itself a
mutation.

Callers who need read concurrency are directed to shard: several caches with
disjoint keyspaces and a deterministic key-to-shard function. Sharding is
documented rather than implemented, because the right shard count depends on
the caller's thread count and the library cannot know it.

## Benchmark methodology

Two synthetic workloads, each run for a fixed number of operations rather than
a fixed wall-clock duration, so that a slower policy is not rewarded with fewer
operations. The steady workload draws keys from a skewed distribution; the scan
workload interleaves the same skewed stream with a sweep over the entire
keyspace.

Each configuration is run five times and the median is reported. The first run
of each configuration is discarded, because it pays the cost of filling a cold
cache and that cost belongs to the cache's first minutes rather than to its
policy.

## Benchmark results

On the steady workload the tiered policy and plain least-recently-used are
within noise of each other, as expected: when the access pattern is stable,
recency and frequency agree about what to keep.

On the scan workload the tiered policy retains the warm working set almost
entirely, while plain least-recently-used retains nearly none of it. The
numbers are invented, and the shape of the result is the whole reason this
record exists.

## Failure modes

A sizer that under-reports causes the resident total to drift below the truth
and the cache to exceed its budget. The consistency check catches this under
test; in production it presents as memory growth that the counters cannot
explain, which is why the check's error message names the sizer.

A caller who pins more than the pinned fraction allows gets an error at pin
time. A caller who pins exactly up to the fraction and then inserts a very
large entry gets an error at insertion time. Both messages name the budget and
both name the pinned total, so the two failures are distinguishable from a log
line alone.

## Observability

`cache.explain()` prints every resolved setting with its origin, every counter,
the resident byte total, and the per-tier occupancy. It is intended to be
pasted into a bug report in full, so it prints absolute numbers rather than
ratios and never truncates.

Nothing is logged by default. A library that writes to a logger the host did
not configure is a library that shows up in the host's error budget for reasons
the host cannot control.

## Testing strategy

The policy is tested against a fake clock and a fake sizer, so that recency,
frequency and size are all directly controllable and no test depends on real
timing or on the memory footprint of a real object. Each test states the
sequence of operations and the expected resident keys, in that order.

The consistency check is itself tested, by a test that deliberately corrupts
the running total and asserts that the check raises. A self-check nobody has
seen fail is a self-check nobody should trust.

## Documentation changes

The caching chapter is rewritten around the budget rather than around the
decorator, with the decorator demoted to an example. The tier vocabulary is
introduced once, with a diagram, and then used consistently; a document that
calls the same tier by two names is a document that teaches nobody anything.

A migration page lists the old parameters, their replacements, and the
translation applied to each. It is written as a table, because a reader
migrating a call site wants to find one row and leave.

## Open questions

Whether the probation fraction should adapt to the observed entry size
distribution rather than being configured. Adaptation would remove a parameter
and add a feedback loop, and feedback loops in eviction policies are
historically the source of oscillation. Left for a later record.

Whether the sidecar should influence admission, so that an entry already on
disk is cheaper to readmit. Plausible, unmeasured, and out of scope here.

## Rejected micro-optimisations

Packing the recency bookkeeping into the entry header to save one allocation
per entry: measured at under one percent on both workloads and rejected as not
worth the readability.

Replacing the intrusive linked list with an array-backed clock approximation:
within noise on both workloads, and it makes the demotion step, which is the
interesting part of this record, substantially harder to explain. Rejected.

Skipping the admission filter for small entries on the theory that they are
cheap to be wrong about: rejected because "small" is another threshold, and the
filter is not the expensive part of an insertion.

## Compatibility matrix

The fictional 4.0 release supports the two previous minor versions of the
serialisation format for the sidecar, reads both, and writes only the newest.
A sidecar file in an unrecognised format is ignored with a single counter
increment rather than an error, because a stale cache file is not a reason to
fail a process start.

The public API gains three keyword parameters and loses one. No positional
parameter changes position, so a caller passing the budget positionally
continues to work, and a caller passing the old maximum entry count
positionally gets the deprecation translation described above.

## Review notes

The reviewer asked why demotion exists at all, given that an entry demoted and
then evicted has merely taken two passes to leave. The answer is the interval
between the two passes: that is where a hit saves a warm entry, and removing
demotion removes the protection that motivated the whole record.

The reviewer also asked for the pinned tier to be dropped as unnecessary. It is
kept because a caller who needs a value present cannot express that with any
combination of the other parameters, and the alternative is that such callers
keep a second cache of their own next to this one.

## Amendment: tier boundaries relaxed

The probation fraction shipped at one fifth of the budget and was relaxed to
one quarter in the fictional 4.1 release, after the invented benchmark showed
that a probation tier smaller than a quarter rejects entries that a second hit
would have justified keeping. The protected tier absorbed the difference and
nothing else changed.

The pinned fraction was left alone. No fictional caller came close to it, and a
cap nobody reaches is a cap that costs nothing to keep.
