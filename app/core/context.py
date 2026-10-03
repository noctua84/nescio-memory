"""Building expanded context regions around search hits.

No DB access here deliberately, mirroring app/core/reassembly.py: this module
takes the hits and the already-fetched chunks and returns regions, so the
planning, hole-splitting, window-resolution, budget and mapping logic is all
testable without a database container. The route owns the two I/O steps
(embed + search, then one range fetch) and nothing else.

See docs/superpowers/specs/2026-10-03-search-context-expansion-design.md. The
rule the whole feature is built to respect: content is reachable only through a
query that semantically matched it. Everything below expands around a hit that
search already returned; nothing here can be asked for a document by path.
"""
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.core.reassembly import join_chunk_run
from app.schemas.search import ContextRegion

ContextMode = Literal["none", "neighbors", "document"]

# A chunk-index span within one document. None is unbounded on that side,
# which is how "the whole document" is expressed.
_Span = tuple[int | None, int | None]

# (repo_name, file_path, from_index, to_index) -- the shape
# LearningRepository.get_chunks_for_ranges takes.
ChunkRange = tuple[str, str, int | None, int | None]

_LOWEST = float("-inf")


@dataclass(frozen=True)
class ContextHit:
    """One search result reduced to what context expansion needs.

    Mapped from the ORM row by the route, so this module never sees a
    `Learning`. `chunk_index` is None when the row's metadata does not carry a
    usable integer index -- such a hit cannot be placed in a document and is
    simply not expanded.
    """

    repo_name: str
    file_path: str
    chunk_index: int | None
    similarity: float


@dataclass(frozen=True)
class FetchedChunk:
    """One row from LearningRepository.get_chunks_for_ranges, as a plain DTO.

    `chunk_size`/`chunk_overlap` are the window this chunk records having been
    cut with, or None for rows ingested before that was recorded.
    """

    repo_name: str
    file_path: str
    chunk_index: int | None
    content: str
    chunk_size: int | None
    chunk_overlap: int | None


def _planned_spans(
    indices: list[int], mode: ContextMode, context_chunks: int
) -> list[_Span]:
    """The index spans `indices` ask for, merged, within one document.

    Overlapping *or adjacent* spans are merged: two runs that touch should be
    one region rather than two with no gap between them.

    The single definition of what a region covers, shared by `plan_ranges` and
    the region builder so the two can never drift -- the fetch widens these
    spans but does not redefine them.
    """
    if mode == "document":
        return [(None, None)] if indices else []
    # Clamped at 0 rather than allowed negative: a negative lower bound would
    # make a region look truncated at a hole below index 0, which is not a hole
    # but the start of the note.
    return _merge_spans(
        [
            (max(0, index - context_chunks), index + context_chunks)
            for index in indices
        ]
    )


def plan_ranges(
    hits: list[ContextHit], mode: ContextMode, context_chunks: int
) -> list[ChunkRange]:
    """Work out the index ranges that have to be fetched for `hits`."""
    if mode == "none":
        return []

    indices_by_document: dict[tuple[str, str], list[int]] = {}
    for hit in hits:
        if hit.chunk_index is None:
            continue
        indices_by_document.setdefault(
            (hit.repo_name, hit.file_path), []
        ).append(hit.chunk_index)

    return [
        # One chunk past the requested upper bound is fetched as a TAIL PROBE
        # and never returned. Invariant 5 (tail completeness) must fire only at
        # the real end of the document, and the only way to know whether a
        # requested region ends there is to ask for the successor: if it comes
        # back, the region is interior and nothing was lost; if it does not,
        # the region is the document's tail and invariant 5 decides. Without
        # the probe the fetched set's highest index IS the region's highest
        # index by construction, every neighbours region looks like the
        # document's tail, and any region ending on a full-size chunk is
        # reported lossy when nothing is missing. (`"document"` needs no probe:
        # it fetches the whole note, so the true highest index is already there.)
        (repo_name, file_path, lo, None if hi is None else hi + 1)
        for (repo_name, file_path), indices in indices_by_document.items()
        for lo, hi in _planned_spans(indices, mode, context_chunks)
    ]


def _merge_spans(spans: list[_Span]) -> list[_Span]:
    merged: list[_Span] = []
    for lo, hi in sorted(spans, key=lambda s: _LOWEST if s[0] is None else s[0]):
        if not merged:
            merged.append((lo, hi))
            continue
        prev_lo, prev_hi = merged[-1]
        # `lo <= prev_hi + 1` rather than `<= prev_hi`: adjacency counts as
        # overlap for merging purposes.
        touches = prev_hi is None or lo is None or lo <= prev_hi + 1
        if touches:
            # prev_lo is already the lower of the two -- the sort put None, and
            # then the smallest index, first.
            new_hi = None if prev_hi is None or hi is None else max(prev_hi, hi)
            merged[-1] = (prev_lo, new_hi)
        else:
            merged.append((lo, hi))
    return merged


def _resolve_window(chunks: list[FetchedChunk]) -> tuple[int, int, bool] | None:
    """Determine the (chunk_size, chunk_overlap) to join `chunks` with.

    The same three-way rule the deleted egress read-back used. `delete_by_file`
    removes every row for a path before re-inserting, so all rows of one
    document come from a single ingest run -- if their recorded windows
    disagree, the rows were written outside the application and the window
    cannot be trusted.

    - Every row agrees on the same valid pair -> use it, window_recorded=True.
      This is the evidence invariant 0 in app/core/reassembly.py needs to trust
      an overlap of 0.
    - No row records either key (rows ingested before the window was recorded)
      -> fall back to the service's current configuration,
      window_recorded=False.
    - Anything else -- rows disagree, or agree on a pair that is partial (one
      key only) or invalid -- returns None, meaning no region for this
      document. Picking a majority would silently reintroduce the bug this
      rule exists to prevent.

    Returns None rather than raising: expansion is additive and must never fail
    the search (see the module docstring's design reference, D2). Resolved over
    every fetched row of the document, not just the ones about to be joined --
    a disagreeing row is evidence about the document, wherever in it it sits.
    """
    distinct_pairs = {(c.chunk_size, c.chunk_overlap) for c in chunks}
    if len(distinct_pairs) != 1:
        return None

    chunk_size, chunk_overlap = next(iter(distinct_pairs))
    if chunk_size is None and chunk_overlap is None:
        # settings is read here, at call time, rather than captured at import,
        # so a reconfigured service is seen -- and so tests can monkeypatch it.
        return settings.chunk_size, settings.chunk_overlap, False
    if (
        isinstance(chunk_size, int)
        and isinstance(chunk_overlap, int)
        and chunk_size > 0
        and 0 <= chunk_overlap < chunk_size
    ):
        return chunk_size, chunk_overlap, True
    return None


def _consecutive_runs(indices: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive integers in `indices` (sorted, distinct)."""
    runs: list[tuple[int, int]] = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


@dataclass
class _Candidate:
    repo_name: str
    file_path: str
    chunk_index_from: int
    chunk_index_to: int
    chunks: list[tuple[int, str]]
    window: tuple[int, int, bool]
    is_document_tail: bool
    truncation_note: str | None
    hit_positions: list[int]
    best_similarity: float


def build_regions(
    hits: list[ContextHit],
    fetched: list[FetchedChunk],
    mode: ContextMode,
    context_chunks: int,
) -> tuple[list[ContextRegion], list[int | None]]:
    """Turn hits plus their fetched chunks into regions and per-hit refs.

    Returns `(contexts, refs)` where `refs[i]` is the `context_ref` for
    `hits[i]` -- an index into `contexts`, or None when no region covers that
    hit.
    """
    if mode == "none":
        return [], [None] * len(hits)

    candidates: list[_Candidate] = []
    for key, document_chunks in _group_by_document(fetched).items():
        candidates.extend(
            _candidates_for_document(key, document_chunks, hits, mode, context_chunks)
        )

    # Rank order, not document order: the budget below is spent on the regions
    # the caller is most likely to want. Ties fall back to the document key so
    # a response is reproducible rather than dict-ordered.
    candidates.sort(
        key=lambda c: (
            -c.best_similarity,
            c.repo_name,
            c.file_path,
            c.chunk_index_from,
        )
    )

    contexts: list[ContextRegion] = []
    refs: list[int | None] = [None] * len(hits)
    budget = settings.max_context_chars
    spent = 0
    omitted = 0

    for candidate in candidates:
        if omitted:
            # Once the budget stopped us we stop building entirely rather than
            # scanning on for a region small enough to squeeze in: that would
            # hand back context in an order the ranking does not justify.
            omitted += 1
            continue

        joined = join_chunk_run(
            candidate.chunks,
            candidate.window[0],
            candidate.window[1],
            window_recorded=candidate.window[2],
            is_document_tail=candidate.is_document_tail,
        )
        length = len(joined.content) if joined.content else 0
        if spent + length > budget:
            omitted = 1
            continue

        spent += length
        notes = [n for n in (joined.reason, candidate.truncation_note) if n]
        contexts.append(
            ContextRegion(
                repo_name=candidate.repo_name,
                file_path=candidate.file_path,
                content=joined.content,
                chunk_index_from=candidate.chunk_index_from,
                chunk_index_to=candidate.chunk_index_to,
                covers=mode,
                exact=joined.exact,
                note="; ".join(notes) if notes else None,
            )
        )
        for position in candidate.hit_positions:
            refs[position] = len(contexts) - 1

    if omitted and contexts:
        # An omitted region is absent from `contexts` by contract, so it has
        # nowhere of its own to carry a note -- yet a caller holding a null
        # `context_ref` needs to know the difference between "that note could
        # not be resolved" and "the response was already full". The last
        # region that did fit is the only place adjacent to the cut, so the
        # explanation goes there. When even the first region overruns the
        # budget there is no such place: `contexts` is empty and every ref is
        # null, which is the one case this cannot explain in-band.
        last = contexts[-1]
        budget_note = (
            f"the {budget}-character context budget was reached after this "
            f"region; {omitted} further region(s) were omitted and the "
            "results referencing them carry context_ref: null"
        )
        contexts[-1] = last.model_copy(
            update={
                "note": f"{last.note}; {budget_note}" if last.note else budget_note
            }
        )

    return contexts, refs


def _group_by_document(
    fetched: list[FetchedChunk],
) -> dict[tuple[str, str], list[FetchedChunk]]:
    grouped: dict[tuple[str, str], list[FetchedChunk]] = {}
    for chunk in fetched:
        grouped.setdefault((chunk.repo_name, chunk.file_path), []).append(chunk)
    return grouped


def _candidates_for_document(
    key: tuple[str, str],
    document_chunks: list[FetchedChunk],
    hits: list[ContextHit],
    mode: ContextMode,
    context_chunks: int,
) -> list[_Candidate]:
    repo_name, file_path = key

    # A row whose meta["chunk_index"] was missing or malformed comes back with
    # chunk_index None (the repository's CASE guard). It cannot be placed in
    # the index sequence, so it is dropped here -- the document still gets a
    # region, just without that chunk, which is the degradation this feature
    # prefers to an error. It stays in the window resolution below, though,
    # because it is still evidence about how this document was ingested.
    window = _resolve_window(document_chunks)
    if window is None:
        return []

    content_by_index = {
        c.chunk_index: c.content for c in document_chunks if c.chunk_index is not None
    }
    if not content_by_index:
        return []
    present = sorted(content_by_index)
    runs = _consecutive_runs(present)
    highest_present = present[-1]

    hit_positions_by_index: dict[int, list[int]] = {}
    for position, hit in enumerate(hits):
        if (
            hit.chunk_index is not None
            and hit.repo_name == repo_name
            and hit.file_path == file_path
        ):
            hit_positions_by_index.setdefault(hit.chunk_index, []).append(position)
    if not hit_positions_by_index:
        return []

    # The spans the hits actually asked for -- NOT the widened ranges that
    # were fetched. The extra chunk `plan_ranges` probes for is present in
    # `content_by_index` (and so in `highest_present`, which is the point) but
    # must not become part of a region.
    planned = _planned_spans(list(hit_positions_by_index), mode, context_chunks)

    candidates: list[_Candidate] = []
    for lo, hi in planned:
        for run_lo, run_hi in runs:
            piece_lo = run_lo if lo is None else max(lo, run_lo)
            piece_hi = run_hi if hi is None else min(hi, run_hi)
            if piece_lo > piece_hi:
                continue

            positions = [
                position
                for index in range(piece_lo, piece_hi + 1)
                for position in hit_positions_by_index.get(index, ())
            ]
            # A piece carrying no hit is text nobody asked about -- it only
            # exists because a planned range reached across a hole into a run
            # whose chunks are all unrequested.
            if not positions:
                continue

            candidates.append(
                _Candidate(
                    repo_name=repo_name,
                    file_path=file_path,
                    chunk_index_from=piece_lo,
                    chunk_index_to=piece_hi,
                    chunks=[
                        (index, content_by_index[index])
                        for index in range(piece_lo, piece_hi + 1)
                    ],
                    window=window,
                    # Invariant 5 (tail completeness) applies only to the real
                    # end of the document. A region ending mid-document has a
                    # successor that is stored and merely outside the window,
                    # which is not a loss.
                    is_document_tail=piece_hi == highest_present,
                    truncation_note=_truncation_note(
                        lo, hi, piece_lo, piece_hi, highest_present
                    ),
                    hit_positions=positions,
                    best_similarity=max(
                        hits[position].similarity for position in positions
                    ),
                )
            )
    return candidates


def _truncation_note(
    lo: int | None,
    hi: int | None,
    piece_lo: int,
    piece_hi: int,
    highest_present: int,
) -> str | None:
    """Say so when a hole, rather than the note's own extent, cut this region.

    Never splice across a hole: the dropped chunk's characters are in no
    neighbouring chunk, so joining would make two passages look adjacent that
    never were. The region is cut instead, and the caller is told.

    The two ends are not symmetrical:

    - Below `piece_lo`, a cut is a hole whenever `piece_lo > 0` -- the chunks
      between 0 and it were never stored. `piece_lo == 0` is the start of the
      note, not a hole.
    - Above `piece_hi`, a cut is a hole only when `piece_hi` is not the
      document's highest stored index. When it is, "the chunk after this one
      was dropped at ingest" and "the note simply ends here" are
      indistinguishable from the stored rows -- that ambiguity is exactly what
      invariant 5 reports through `exact`, so repeating it as a truncation note
      would turn every short note into a warning.
    """
    cuts = []
    if (lo is None or lo < piece_lo) and piece_lo > 0:
        cuts.append(f"below chunk {piece_lo}")
    if (hi is None or hi > piece_hi) and piece_hi < highest_present:
        cuts.append(f"above chunk {piece_hi}")
    if not cuts:
        return None
    return (
        "truncated " + " and ".join(cuts) + ": a chunk was dropped at ingest, so "
        "the region stops at the hole rather than splicing across it"
    )
