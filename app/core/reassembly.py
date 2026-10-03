"""Pure joining of stored chunks back into contiguous text.

No DB access here deliberately -- the repository fetches ordered rows, this
module only reasons about the (chunk_index, content) pairs. That split is
what let the algorithm be verified in isolation: every document length
0..6000 was run through `reassemble` with zero false `exact` claims.

`chunk_text` (app/core/chunking.py) emits `text[i*step : i*step + size]` for
`i*step < len(text)`, with `step = size - overlap`. Ingest enumerates ALL
chunks from that pre-filter `enumerate` before dropping any chunk whose
stripped length is under 50 characters, so `chunk_index` is the chunk's
original position and a dropped chunk leaves a visible hole in the stored
index sequence rather than silently shifting everything after it.

Two entry points, because there are two questions:

- `join_chunk_run` joins any contiguous RUN of chunks. A run may start at a
  non-zero index and may end on a full-size chunk, because a mid-document
  region's successor exists and is simply outside the requested window rather
  than lost. This is what search context expansion calls (see
  docs/superpowers/specs/2026-10-03-search-context-expansion-design.md).
- `reassemble` answers the whole-document question: it additionally requires
  the run to be the entire document (indices exactly 0..n-1) and the last
  chunk to be the document's real tail.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class JoinResult:
    """Outcome of joining one contiguous run of chunks.

    `content` is the best-effort text whenever one exists, and None when no
    safe join can be formed at all -- an empty run, or a hole in the run's
    index sequence, where the characters the dropped chunk covered are in no
    neighbouring chunk, so joining across it would splice together two parts
    of the document that were never adjacent. That is precisely the
    silently-wrong output this module exists to refuse, so nothing is returned
    rather than something plausible.

    The remaining refusals (coverage gap, overlap mismatch, dropped trailing
    chunk) do carry partial text: in each of those the prefix assembled so far
    is genuinely a prefix of the original.

    `reason` is always set when `exact` is False, and is the only thing that
    distinguishes the cases -- do not infer the cause from `content is None`.
    """

    content: str | None
    exact: bool
    reason: str | None


@dataclass(frozen=True)
class ReassemblyResult:
    """Outcome of attempting to reassemble a whole document from its chunks.

    `content` / `exact` / `reason` carry the same meaning as on JoinResult;
    `chunk_count` is how many stored chunks were supplied.
    """

    content: str | None
    exact: bool
    reason: str | None
    chunk_count: int


def _redundancy_refusal(
    chunk_size: int, chunk_overlap: int, chunk_count: int, window_recorded: bool
) -> str | None:
    """Invariant 0: there has to be redundancy to verify -- UNLESS the window
    itself is a recorded fact rather than a guess.

    Overlap agreement (invariant 4) is the only evidence that the configured
    window is the one these chunks were actually cut with. With
    chunk_overlap == 0 there is no shared region at all, so a multi-chunk run
    carries no such evidence and plain concatenation cannot be distinguished
    from a join that duplicates a real overlap at every seam. Measured: a
    corpus ingested at 1000/200 and read back at 1000/0 returns the wrong text
    on 69% of document lengths, and every invariant below passes, because each
    stored chunk is exactly step == 1000 long.

    That reasoning holds only when chunk_size/chunk_overlap are a guess about
    the service's current configuration (window_recorded=False) -- "the chunks
    were genuinely cut at overlap 0" and "the service has since been
    reconfigured to overlap 0" are then indistinguishable from the stored data,
    and claiming `exact` would be claiming something unknowable. When the
    window is instead read from the rows' own recorded metadata
    (window_recorded=True), an overlap of 0 is not a guess -- it is the window
    the chunks were actually cut with, the rows say so, and plain concatenation
    is then the correct and provable join. Refusing it anyway would reject a
    provably correct answer for lack of evidence that already exists elsewhere.

    A single chunk is unaffected either way: with nothing to join there is no
    overlap to duplicate, and the content is the run.

    Shared by both entry points so the two can never disagree about when
    exactness is unprovable, or about the wording of the refusal.
    """
    if window_recorded or chunk_overlap != 0 or chunk_count <= 1:
        return None
    return (
        "chunk_overlap is 0, so the stored chunks share no region that "
        "could confirm they were cut with the configured "
        f"chunk_size={chunk_size}/chunk_overlap={chunk_overlap}; "
        f"exactness is not provable for a {chunk_count}-chunk document "
        "without that evidence"
    )


def join_chunk_run(
    chunks: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
    *,
    window_recorded: bool,
    is_document_tail: bool,
) -> JoinResult:
    """Join one contiguous run of `(chunk_index, content)` pairs into a string.

    `chunks` must already be ordered ascending by chunk_index. The run's first
    index is whatever it is: contiguity is checked relative to that index, not
    against 0, because a context region is a sub-range of a document and a
    mid-document region legitimately starts at index 4.

    `window_recorded` states whether `chunk_size`/`chunk_overlap` were read
    from the stored rows' own metadata (True) or are a guess based on the
    service's current configuration (False, for rows ingested before the
    window was recorded per row). Required, with no default, so a caller
    cannot pass the window without also stating how much to trust it -- that
    is exactly the distinction invariant 0 needs.

    `is_document_tail` states whether the run's last chunk is the highest
    index stored for its document. Invariant 5 below is applied only when it
    is: for a mid-document run, a full-size last chunk means its successor
    exists and was simply not requested, which is not a loss and must not be
    reported as one.
    """
    chunk_count = len(chunks)

    if not chunks:
        return JoinResult(content=None, exact=False, reason="no chunks to join")

    step = chunk_size - chunk_overlap

    redundancy = _redundancy_refusal(
        chunk_size, chunk_overlap, chunk_count, window_recorded
    )
    if redundancy is not None:
        return JoinResult(
            content="".join(content for _, content in chunks),
            exact=False,
            reason=redundancy,
        )

    # Invariant 1 (run-relative contiguity). The stored indices must be
    # consecutive from the run's own first index -- a hole means a chunk was
    # dropped by the 50-character minimum, and the text it covered is gone for
    # good (not recoverable from neighbors). Whether the run starts at 0 is a
    # separate question, asked only by `reassemble` below.
    indices = [idx for idx, _ in chunks]
    first = indices[0]
    expected = list(range(first, first + chunk_count))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        return JoinResult(
            content=None,
            exact=False,
            reason=(
                f"chunk indices are not contiguous from {first}: missing "
                f"{missing}"
                if missing
                else (
                    f"chunk indices are not contiguous from {first}: got "
                    f"{indices}"
                )
            ),
        )

    contents = [content for _, content in chunks]
    out = contents[0]

    for k in range(1, chunk_count):
        prev = contents[k - 1]
        curr = contents[k]
        o = len(prev) - step

        # Invariant 2: coverage. A negative shared-region length means prev
        # ended before curr began -- chunks that do not touch cannot be joined.
        if o < 0:
            return JoinResult(
                content=out,
                exact=False,
                reason=(
                    f"coverage gap: chunk {k - 1} ends before chunk {k} begins"
                ),
            )

        # Invariant 3: there must be a shared region to verify at all.
        #
        # With chunk_overlap > 0, o == 0 makes invariant 4 below vacuous --
        # prev[step:] and curr[:0] are both "" and compare equal no matter what
        # the chunks contain -- and `out += curr[o:]` then appends curr whole,
        # including the overlap it shares with prev. Every join silently
        # duplicates chunk_overlap characters.
        #
        # This is reachable from configuration alone, with no tampering: a
        # corpus ingested at 1000/200 and read back at CHUNK_OVERLAP=0 (legal
        # per app/config.py, which only requires 0 <= overlap < size) gives
        # step == 1000 == the stored chunk length, so o == 0 at every join. A
        # 2,600-character document came back as 3,200 characters reporting
        # `exact: true`; swept over lengths 50..6000 it was wrong on 69% of
        # them. 1200/200 and 2000/1000 are the same defect.
        #
        # Refusing is sound rather than merely cautious: at a MATCHED window
        # with overlap > 0, a non-final chunk can never have len(prev) == step.
        # len(prev) == step would mean the document ends exactly at the start of
        # the next window, so chunk_text's `while start < len(text)` would never
        # have emitted curr at all -- prev would be the final chunk of the
        # document, and a document's final chunk is never used as `prev` within
        # a run. So o == 0 here always means the window parameters are not the
        # ones the chunks were cut with.
        #
        # Guarded on chunk_overlap > 0 because with overlap == 0 there is
        # genuinely no shared region: o == 0 is correct for every full chunk and
        # plain concatenation is the right join.
        if chunk_overlap > 0 and o == 0:
            return JoinResult(
                content=out,
                exact=False,
                reason=(
                    f"chunk {k - 1} is exactly step={step} characters long, so "
                    "there is no overlap to verify and no evidence the "
                    f"configured chunk_size={chunk_size}/chunk_overlap="
                    f"{chunk_overlap} are the ones these chunks were cut with; "
                    "refusing rather than returning a reassembly that would "
                    "duplicate the overlap at every join"
                ),
            )

        # o > chunk_overlap means prev is longer than chunk_size, which chunk_text
        # never produces for a correctly-configured window -- treat it the same
        # as a mismatch between the stored window parameters and the current ones.
        if o > chunk_overlap:
            return JoinResult(
                content=out,
                exact=False,
                reason=(
                    f"chunk {k - 1} is longer than chunk_size={chunk_size}, "
                    "which chunk_text never produces -- the configured "
                    "chunk_size/chunk_overlap likely changed since ingest"
                ),
            )

        # Invariant 4: overlap agreement. The shared region must be the same
        # substring of the original document in both chunks. This is also what
        # catches a CHUNK_SIZE/CHUNK_OVERLAP change made after ingest for rows
        # that do not record the window they were cut with.
        if prev[step:] != curr[:o]:
            return JoinResult(
                content=out, exact=False, reason=f"overlap mismatch at index {k}"
            )

        out += curr[o:]

    # Invariant 5: tail completeness. If the last chunk of the DOCUMENT is full
    # size, a successor chunk existed (chunk_text only emits a short final
    # chunk) and was dropped by the 50-character minimum -- the document does
    # not end where the stored chunks end. Deliberately pessimistic: a dropped
    # trailing chunk is sometimes wholly contained in its predecessor, but the
    # stored rows don't record len(text), so that case can't be distinguished
    # from real truncation. Refusing both is the only honest option.
    #
    # Applied only when this run IS the document's tail. For a mid-document
    # run, a full-size last chunk is the normal case -- its successor is stored
    # and merely outside the requested window -- so firing here would report
    # every interior context region as lossy when nothing was lost at all.
    if is_document_tail and len(contents[-1]) >= chunk_size:
        return JoinResult(
            content=out,
            exact=False,
            reason=(
                "last stored chunk is full size, so a successor chunk existed "
                "and was dropped by the 50-character minimum"
            ),
        )

    return JoinResult(content=out, exact=True, reason=None)


def reassemble(
    chunks: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
    *,
    window_recorded: bool,
) -> ReassemblyResult:
    """Reassemble a WHOLE document from `chunks` (already ordered by index).

    A thin wrapper over `join_chunk_run`: it adds the two requirements that
    only make sense for a whole document -- the stored indices must be exactly
    0..n-1, and the run's last chunk is by definition the document's tail, so
    invariant 5 applies -- and delegates the joining itself.

    See `join_chunk_run` for the meaning of `window_recorded` and for the
    invariants. `exact` is only ever True when all of them hold.
    """
    chunk_count = len(chunks)

    if not chunks:
        return ReassemblyResult(
            content=None, exact=False, reason="no chunks stored", chunk_count=0
        )

    # Invariant 0 is checked here as well as inside join_chunk_run so that it
    # is reported ahead of the 0..n-1 check below: a run with both problems is
    # unprovable for the more fundamental reason, and that ordering is what
    # callers of this function have always seen.
    redundancy = _redundancy_refusal(
        chunk_size, chunk_overlap, chunk_count, window_recorded
    )
    if redundancy is not None:
        return ReassemblyResult(
            content="".join(content for _, content in chunks),
            exact=False,
            reason=redundancy,
            chunk_count=chunk_count,
        )

    # Whole-document contiguity: the indices must be exactly 0..n-1. A run
    # starting above 0 means the document's opening chunks are missing, which
    # join_chunk_run deliberately permits (a context region is a sub-range) and
    # a document read-back must not.
    indices = [idx for idx, _ in chunks]
    expected = list(range(chunk_count))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        return ReassemblyResult(
            content=None,
            exact=False,
            reason=(
                "chunk indices are not contiguous from 0: missing "
                f"{missing}"
                if missing
                else f"chunk indices are not contiguous from 0: got {indices}"
            ),
            chunk_count=chunk_count,
        )

    joined = join_chunk_run(
        chunks,
        chunk_size,
        chunk_overlap,
        window_recorded=window_recorded,
        is_document_tail=True,
    )
    return ReassemblyResult(
        content=joined.content,
        exact=joined.exact,
        reason=joined.reason,
        chunk_count=chunk_count,
    )
