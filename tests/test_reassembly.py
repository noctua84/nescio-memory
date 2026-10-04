"""Chunk joining in isolation, with no database in the way.

`app/core/reassembly.py` is retrieval infrastructure: it is what turns the
positional 1000/200 windows the corpus is stored as back into readable passages,
by stripping the overlap consecutive chunks share. Everything that reads stored
content goes through it -- today that is context expansion in
`POST /api/v1/search` (see
docs/superpowers/specs/2026-10-03-search-context-expansion-design.md, D3).

Concatenating chunks instead would duplicate `chunk_overlap` characters at every
seam, which is wrong on the page and actively harmful in a prompt. The one
promise this module makes is the implication everything else rests on:

    result.exact is True   =>   result.content == the corresponding original text

For `reassemble`, "corresponding original text" is the whole document. For
`join_chunk_run` -- the shared core, and what context expansion actually calls
-- it is the span the run covers, which legitimately starts mid-document. A
refusal that is merely pessimistic costs a caller some context. A false `exact`
puts a passage in a model's prompt with characters silently missing from the
middle, so the implication is swept over hundreds of document lengths below
rather than argued for in a comment.

The rest of the file is one test per refusal, each violating exactly one
invariant and asserting both the reason and whether `content` is None or a
best-effort partial -- the callers distinguish those two cases.
"""
import pytest

from app.api.v1.ingest import MIN_CONTENT_CHARS
from app.config import settings
from app.core.chunking import chunk_text
from app.core.reassembly import join_chunk_run, reassemble

# The window the stored corpus was produced with. Read from settings rather
# than hardcoded so that a changed default breaks these tests loudly instead of
# leaving them asserting against a window nothing uses.
SIZE = settings.chunk_size
OVERLAP = settings.chunk_overlap
STEP = SIZE - OVERLAP

_SENTENCE = (
    "Recoverability has to exist before the corpus moves, which is why the "
    "egress path lands ahead of the other known gaps. "
)


def document_of_length(length: int) -> str:
    """A readable, non-repeating document of exactly `length` characters.

    Non-repeating matters: every sentence carries its own ordinal, so no two
    `STEP`-aligned windows of the text are equal. Against a document of
    `"a" * n` the overlap-agreement invariant would pass no matter how the
    window parameters were mangled, and the refusal tests would prove nothing.

    The first and last characters are forced to be non-whitespace so that a
    chunk's stripped length equals its real length at the document's edges.
    That is what makes the length expectations in the sweep below exact rather
    than approximate: without it, a document truncated mid-space could lose its
    only chunk to ingest's 50-character minimum at a length the test expects to
    survive.

    tests/test_search_context.py imports this builder rather than growing its
    own, so "a well-formed document" has one definition across the suite.
    """
    if length <= 0:
        return ""

    parts: list[str] = []
    total = 0
    ordinal = 0
    while total < length:
        part = f"Sentence {ordinal:05d}: {_SENTENCE}"
        parts.append(part)
        total += len(part)
        ordinal += 1

    characters = list("".join(parts)[:length])
    if characters[0].isspace():
        characters[0] = "A"
    if characters[-1].isspace():
        characters[-1] = "Z"
    return "".join(characters)


def store_as_ingest_would(text: str) -> list[tuple[int, str]]:
    """Chunk `text` and filter it exactly the way app/api/v1/ingest.py does.

    The fidelity of this function is the whole point of the file: ingest
    enumerates ALL chunks and stores `chunk_index = i` from that PRE-filter
    enumeration, then skips any chunk whose stripped length is under
    MIN_CONTENT_CHARS. A dropped chunk therefore leaves a hole in the stored
    index sequence instead of silently renumbering its successors, and that
    hole is what invariant 1 detects. Filtering before enumerating here would
    make every test in this file agree with a reassembler that cannot exist.
    """
    return [
        (index, chunk)
        for index, chunk in enumerate(chunk_text(text, SIZE, OVERLAP))
        if len(chunk.strip()) >= MIN_CONTENT_CHARS
    ]


# ---------------------------------------------------------------------------
# The exact round trip, over several document shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, length",
    [
        # Below SIZE, so chunk_text emits exactly one window and there is no
        # overlap arithmetic to get wrong.
        ("single chunk", 300),
        # In (STEP, 2 * STEP], so exactly two windows with a real shared region.
        ("two chunks", STEP + 400),
        # Several interior full-size chunks plus a short tail.
        ("many chunks", 5 * STEP + 250),
        # An exact multiple of the step: the arithmetic edge where the final
        # window starts exactly where the previous one's shared region ends.
        ("exact multiple of step", 3 * STEP),
        # One character past a step boundary, so the final window is a single
        # character short of a full one.
        ("one past a step boundary", 2 * STEP + 1),
    ],
)
@pytest.mark.parametrize("window_recorded", [True, False])
def test_a_well_formed_document_round_trips_exactly(shape, length, window_recorded):
    """Both window provenances, because at OVERLAP > 0 neither may differ.

    `window_recorded` exists for invariant 0, which only fires at an overlap of
    0. At the configured window there is a real shared region to verify, so the
    verdict must not depend on whether the window was recorded or guessed --
    asserting that here is what keeps invariant 0 from quietly widening into a
    refusal of ordinary legacy documents.
    """
    text = document_of_length(length)

    result = reassemble(
        store_as_ingest_would(text), SIZE, OVERLAP, window_recorded=window_recorded
    )

    assert result.exact is True, f"{shape}: refused with {result.reason!r}"
    assert result.reason is None
    # Not "starts with", not "same length": the recovered document must be the
    # byte-for-byte original, because a caller writes this straight to disk.
    assert result.content == text


def test_a_unicode_document_round_trips_exactly():
    """Proves the window slices characters, not bytes.

    chunk_text slices a `str`, so a multi-byte document is only recovered
    exactly if every index in the reassembly arithmetic is a character index.
    A byte-oriented implementation would either split a codepoint or compute a
    shared region of the wrong width, and the assertion below would fail with
    mangled text rather than with a refusal.
    """
    text = ("Übergrößenträger – αβγδε – 日本語のテキスト – naïve café façade. " * 40).strip()

    # Guard the premise: if this stopped being multi-byte the test would still
    # pass while proving nothing.
    assert len(text.encode("utf-8")) > len(text)
    assert len(text) > SIZE, "the document must span more than one window"

    result = reassemble(
        store_as_ingest_would(text), SIZE, OVERLAP, window_recorded=True
    )

    assert result.exact is True, f"refused with {result.reason!r}"
    assert result.content == text


# ---------------------------------------------------------------------------
# The central safety property, swept over document lengths
# ---------------------------------------------------------------------------


def _sweep_lengths() -> list[int]:
    """Document lengths dense where the arithmetic changes shape.

    Every short length (the region where ingest's 50-character minimum decides
    whether anything is stored at all), a stride across the 0..6000 range the
    design claims to have verified, and the immediate neighbourhood of every
    step and window boundary, where an off-by-one in the shared-region width
    would live.
    """
    lengths = set(range(0, 130))
    lengths.update(range(0, 6001, 31))
    for boundary in list(range(STEP, 6001, STEP)) + list(range(SIZE, 6001, SIZE)):
        lengths.update(range(boundary - 2, boundary + 3))
    return sorted(length for length in lengths if 0 <= length <= 6000)


@pytest.mark.parametrize("length", _sweep_lengths())
def test_reassembly_never_claims_exact_for_anything_but_the_original(length):
    """`exact is True` implies `content == original`, at every length.

    This is the property the egress design rests on, so it is a test rather
    than a sentence in a spec. The implication is asserted first and
    unconditionally -- it must survive any change to the document builder.

    The two concrete expectations after it are stronger, and hold for prose
    (no long whitespace runs): below the 50-character minimum nothing is
    stored, and at or above it the document always comes back exactly. The
    second is not obvious -- it covers the case where the short final chunk is
    dropped by the minimum, which is safe only because that chunk's
    predecessor still reaches the end of the document.
    """
    text = document_of_length(length)
    stored = store_as_ingest_would(text)

    # window_recorded=True is the production path: ingest stamps chunk_size and
    # chunk_overlap into every row, so read-back reassembles with the window the
    # chunks were provably cut with rather than whatever settings says now.
    result = reassemble(stored, SIZE, OVERLAP, window_recorded=True)

    if result.exact:
        assert result.content == text, (
            f"length {length}: claimed exact but returned "
            f"{len(result.content or '')} characters that differ from the input"
        )

    if length < MIN_CONTENT_CHARS:
        # Ingest drops every chunk of such a file, so there is nothing to
        # assemble. The route turns this into a 404, not a 409.
        assert stored == []
        assert result.exact is False
        assert result.content is None
        assert result.chunk_count == 0
        assert result.reason == "no chunks stored"
    else:
        assert result.exact is True, f"length {length}: refused with {result.reason!r}"
        assert result.content == text


@pytest.mark.parametrize("length", _sweep_lengths())
def test_a_legacy_documents_verdict_matches_a_recorded_ones_at_the_real_window(length):
    """A legacy row's verdict must be identical, at the window it was cut with.

    Rows ingested before the window was recorded per row reassemble with
    `window_recorded=False`, which switches invariant 0 on. Invariant 0 is a
    refusal of a *guessed* zero overlap, so at the configured window -- where
    the overlap is 200 and the guess happens to be right -- it must not fire,
    and the whole verdict must be indistinguishable from the recorded case.

    This is the regression guard on the fix's blast radius: 424 legacy
    documents read back after an upgrade must behave exactly as they did
    before, and a sweep comparing the two flags length by length is what says
    so. The `exact is True` assertion is kept rather than only comparing, so
    the pair cannot agree on being wrong.
    """
    text = document_of_length(length)
    stored = store_as_ingest_would(text)

    guessed = reassemble(stored, SIZE, OVERLAP, window_recorded=False)
    recorded = reassemble(stored, SIZE, OVERLAP, window_recorded=True)

    assert guessed == recorded, (
        f"length {length}: window provenance changed the verdict at a matched "
        f"window ({guessed.reason!r} vs {recorded.reason!r})"
    )
    if length >= MIN_CONTENT_CHARS:
        assert guessed.exact is True, f"length {length}: refused with {guessed.reason!r}"
        assert guessed.content == text


# ---------------------------------------------------------------------------
# One refusal per invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_recorded", [True, False])
def test_no_chunks_at_all_is_refused_with_no_content(window_recorded):
    """Short-circuits ahead of invariant 0, so provenance cannot matter."""
    result = reassemble([], SIZE, OVERLAP, window_recorded=window_recorded)

    assert result.exact is False
    assert result.content is None
    assert result.chunk_count == 0
    assert result.reason == "no chunks stored"


def test_a_hole_in_the_index_sequence_is_refused_with_no_content():
    """Invariant 1, contiguity.

    The characters the dropped chunk covered exist in no neighbouring chunk, so
    joining across the hole would splice together two regions of the document
    that were never adjacent. `content` is None rather than a best-effort
    splice: that splice is exactly the silently-wrong output this module
    refuses to produce.
    """
    text = document_of_length(3 * STEP)
    stored = store_as_ingest_would(text)
    assert len(stored) == 3, "the fixture must be a three-chunk document"

    with_a_hole = [stored[0], stored[2]]

    result = reassemble(with_a_hole, SIZE, OVERLAP, window_recorded=True)

    assert result.exact is False
    assert result.content is None, "a hole must not be papered over with a splice"
    assert result.chunk_count == 2
    assert "not contiguous" in result.reason
    # The reason names the index that is gone, which is what makes a 409 on
    # this route actionable rather than merely discouraging.
    assert "missing [1]" in result.reason


def test_chunks_out_of_index_order_are_refused():
    """Also invariant 1, via its other branch.

    `reassemble` documents that it receives rows already ordered by
    chunk_index. If that ever stopped being true -- a changed ORDER BY, a
    chunk_index that no longer casts to an integer -- the indices would be a
    permutation of 0..n-1 with nothing missing, and the text would be joined in
    the wrong order. That branch reports the sequence it actually got.
    """
    text = document_of_length(2 * STEP)
    stored = store_as_ingest_would(text)
    assert len(stored) == 2

    result = reassemble([stored[1], stored[0]], SIZE, OVERLAP, window_recorded=True)

    assert result.exact is False
    assert result.content is None
    assert "not contiguous" in result.reason
    assert "got [1, 0]" in result.reason


def test_chunks_that_do_not_touch_are_refused_with_a_partial():
    """Invariant 2, coverage.

    Two short chunks stored under contiguous indices cannot have come from one
    sliding window: a chunk shorter than the step ended before its successor
    began, so there is no shared region to align on and an unknown number of
    characters between them is missing. The prefix assembled so far is still a
    genuine prefix of the original, so it is carried on the result -- a context
    region returns it with `exact: false` and a `note`, rather than returning
    nothing at all.
    """
    first = "first fragment, far shorter than the step this window advances by"
    second = "second fragment, which begins somewhere well past where the first ended"
    assert len(first) < STEP

    result = reassemble([(0, first), (1, second)], SIZE, OVERLAP, window_recorded=True)

    assert result.exact is False
    assert result.content == first, "the prefix before the gap is still usable"
    assert result.chunk_count == 2
    assert result.reason == "coverage gap: chunk 0 ends before chunk 1 begins"


def test_a_disagreeing_shared_region_is_refused_with_a_partial():
    """Invariant 3, overlap agreement.

    The region two consecutive chunks share is the same substring of the
    original in both of them, so it must match character for character. Here
    the second chunk's opening does not match the first chunk's tail at all,
    which is what a chunk belonging to a different document -- or a window
    reconfigured since ingest -- looks like.
    """
    result = reassemble(
        [(0, "a" * SIZE), (1, "b" * 300)], SIZE, OVERLAP, window_recorded=True
    )

    assert result.exact is False
    assert result.content == "a" * SIZE
    assert result.reason == "overlap mismatch at index 1"


def test_a_window_reconfigured_since_ingest_is_refused_rather_than_mangled():
    """Invariant 3 again, in the form that actually happens in production.

    The window parameters are not recorded per row, so a service restarted with
    a different CHUNK_OVERLAP would compute a shared region of the wrong width
    and happily emit a document with a few hundred characters duplicated or
    dropped at every join. Overlap agreement is the only thing standing in the
    way, so this pins the realistic case: chunks produced at SIZE/OVERLAP,
    reassembled under a narrower overlap.
    """
    text = document_of_length(3 * STEP)
    stored = store_as_ingest_would(text)
    reconfigured_overlap = OVERLAP // 2

    result = reassemble(stored, SIZE, reconfigured_overlap, window_recorded=False)

    assert result.exact is False
    assert result.reason is not None
    # Whichever of the two reconfiguration branches fires, the one thing that
    # must never happen is a confident wrong answer.
    assert result.content != text


def test_a_chunk_longer_than_the_window_is_refused_as_a_reconfiguration():
    """Invariant 3's sibling guard: a chunk longer than chunk_size.

    chunk_text never emits a window wider than chunk_size, so a stored chunk
    that is wider means the configured window shrank since ingest. Caught
    explicitly rather than left to the overlap comparison, because a shared
    region wider than chunk_overlap means the two chunks' alignment is not
    merely wrong but undefined.
    """
    oversized = "x" * (SIZE + 200)

    result = reassemble(
        [(0, oversized), (1, "y" * 300)], SIZE, OVERLAP, window_recorded=False
    )

    assert result.exact is False
    assert result.content == oversized
    assert f"longer than chunk_size={SIZE}" in result.reason
    assert "likely changed since ingest" in result.reason


def test_a_full_size_final_chunk_is_refused_with_a_partial():
    """Invariant 4, tail completeness.

    chunk_text only ever emits a SHORT final window, so a full-size chunk at
    the highest stored index proves a successor existed and was dropped by the
    50-character minimum: the document does not end where the stored chunks
    end. Deliberately pessimistic -- a dropped trailing chunk is sometimes
    wholly contained in its predecessor, but the rows do not record the
    original length, so real truncation cannot be told apart from it. What is
    returned is a true prefix.
    """
    only_chunk = "t" * SIZE

    result = reassemble([(0, only_chunk)], SIZE, OVERLAP, window_recorded=True)

    assert result.exact is False
    assert result.content == only_chunk
    assert result.chunk_count == 1
    assert "a successor chunk existed and was dropped" in result.reason


def test_a_dropped_trailing_chunk_from_real_chunking_is_refused():
    """Invariant 4 reached through chunk_text rather than a hand-built list.

    At the production 1000/200 window the arithmetic makes this unreachable:
    when a tail short enough to be dropped exists, its predecessor is always
    short of full size. At a narrower window it is reachable, and this pins
    that the refusal fires from genuinely chunked input and not only from a
    constructed one.
    """
    size, overlap = 100, 10
    step = size - overlap
    text = document_of_length(120)

    stored = [
        (index, chunk)
        for index, chunk in enumerate(chunk_text(text, size, overlap))
        if len(chunk.strip()) >= MIN_CONTENT_CHARS
    ]
    # Chunk 1 is text[90:190] -> 30 characters, under the minimum, so it is
    # dropped and chunk 0 (a full 100 characters) becomes the highest index.
    assert stored == [(0, text[:size])]
    assert len(text) > step + 1, "the dropped tail must really have existed"

    result = reassemble(stored, size, overlap, window_recorded=True)

    assert result.exact is False
    assert result.content == text[:size]
    assert result.content != text, "the tail really is missing from the output"
    assert "a successor chunk existed and was dropped" in result.reason


# ---------------------------------------------------------------------------
# Invariant 0: no redundancy to verify, unless the window is a recorded fact
# ---------------------------------------------------------------------------


def store_at_window(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """store_as_ingest_would, at a window other than the configured one."""
    return [
        (index, chunk)
        for index, chunk in enumerate(chunk_text(text, size, overlap))
        if len(chunk.strip()) >= MIN_CONTENT_CHARS
    ]


def test_chunk_text_honours_an_explicit_zero_overlap():
    """Guards the premise of every invariant-0 test below.

    `chunk_text` used to resolve its arguments with `or`, which turned an
    explicit `overlap=0` into the configured default -- so a test asking for
    zero overlap silently got 200 and proved nothing. That is fixed, and the
    fix is load-bearing here.
    """
    text = document_of_length(3 * SIZE)

    chunks = chunk_text(text, SIZE, 0)

    assert len(chunks) == 3
    # Non-overlapping windows tile the document exactly, which is what makes
    # plain concatenation the correct reassembly at overlap 0.
    assert "".join(chunks) == text
    assert chunks[0] == text[:SIZE]
    assert chunks[1] == text[SIZE : 2 * SIZE]


def test_a_guessed_zero_overlap_refuses_a_multi_chunk_document():
    """Invariant 0.

    With `chunk_overlap == 0` the stored chunks share no region, so nothing in
    the data can confirm they were cut at that window. "These chunks were
    genuinely cut at overlap 0" and "the service has since been reconfigured
    to overlap 0" are indistinguishable, and the second is the audited bug: a
    corpus cut at 1000/200 and read at 1000/0 came back wrong on 69% of
    document lengths with `exact: true`.

    Note what is asserted about `content`: the concatenation IS the right
    answer in this particular case, because these chunks really were cut at
    overlap 0. The refusal is not a claim that the text is wrong -- it is a
    refusal to *claim* it is right on evidence that does not exist. That
    distinction is why the partial text is still carried.
    """
    text = document_of_length(2600)
    stored = store_at_window(text, SIZE, 0)
    assert len(stored) > 1, "invariant 0 only applies to multi-chunk documents"

    result = reassemble(stored, SIZE, 0, window_recorded=False)

    assert result.exact is False
    assert result.reason is not None
    assert "chunk_overlap is 0" in result.reason
    assert "exactness is not provable" in result.reason
    assert result.chunk_count == len(stored)
    assert result.content == text  # correct, but not provably so


def test_a_recorded_zero_overlap_reassembles_exactly():
    """The other half of invariant 0, and the reason it is conditional.

    When the rows themselves record `chunk_overlap: 0`, an overlap of 0 is not
    a guess -- it is the window these chunks were cut with, and plain
    concatenation is then provably correct. Refusing it anyway would reject a
    correct answer for want of evidence that already exists in the row.
    """
    text = document_of_length(2600)
    stored = store_at_window(text, SIZE, 0)

    result = reassemble(stored, SIZE, 0, window_recorded=True)

    assert result.exact is True, f"refused with {result.reason!r}"
    assert result.reason is None
    assert result.content == text


def test_a_single_chunk_at_zero_overlap_is_exact_whatever_the_provenance():
    """Invariant 0 is scoped to `chunk_count > 1`, deliberately.

    With nothing to join there is no overlap that could be duplicated, so the
    stored chunk is the document and a guessed window cannot corrupt anything.
    """
    text = document_of_length(600)
    stored = store_at_window(text, SIZE, 0)
    assert len(stored) == 1

    for window_recorded in (True, False):
        result = reassemble(stored, SIZE, 0, window_recorded=window_recorded)
        assert result.exact is True, f"refused with {result.reason!r}"
        assert result.content == text


# ---------------------------------------------------------------------------
# Invariant 3a: a shared region of zero width makes invariant 3 vacuous
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "read_size, read_overlap",
    [
        # Both give step == SIZE == the stored chunk length, so o == 0 at every
        # join while chunk_overlap is still > 0 -- the exact shape invariant 3a
        # exists for.
        (1200, 200),
        (2000, 1000),
    ],
)
def test_a_zero_width_shared_region_is_refused(read_size, read_overlap):
    """Invariant 3a.

    At `o == 0`, invariant 3 compares `prev[step:]` with `curr[:0]` -- both
    empty, equal no matter what the chunks contain -- and `out += curr[o:]`
    then appends `curr` whole, including the characters it shares with `prev`.
    Every join silently duplicates the overlap, and the old code reported
    `exact: true` for it.

    It is sound to refuse rather than merely cautious: at a matched window with
    overlap > 0, a non-final chunk can never have `len(prev) == step`, because
    that would mean the document ended where the next window starts and
    `chunk_text` would not have emitted a successor at all. So `o == 0` here
    always means the window is not the one the chunks were cut with.
    """
    text = document_of_length(2600)
    stored = store_as_ingest_would(text)  # cut at the configured SIZE/OVERLAP
    assert len(stored) > 1
    assert (
        read_size - read_overlap == SIZE
    ), "the fixture requires a read-time step equal to the stored chunk length"

    result = reassemble(stored, read_size, read_overlap, window_recorded=False)

    assert result.exact is False
    assert result.reason is not None
    assert f"exactly step={SIZE} characters long" in result.reason
    assert "no overlap to verify" in result.reason
    # A strict prefix is still carried on the result, so a region can return
    # the part that is trustworthy -- and it is emphatically not the mangled
    # whole-document text an unguarded join produces.
    assert result.content == stored[0][1]
    assert result.content != text


def test_invariant_3a_does_not_fire_at_a_matched_window():
    """The scoping guard: `o == 0` is normal at overlap 0, and absent above it.

    Invariant 3a is conditioned on `chunk_overlap > 0`. Without that condition
    it would refuse every legitimate zero-overlap document, where `o == 0` is
    correct for every full chunk.
    """
    text = document_of_length(2600)

    at_zero = store_at_window(text, SIZE, 0)
    assert reassemble(at_zero, SIZE, 0, window_recorded=True).exact is True

    at_configured = store_as_ingest_would(text)
    assert (
        reassemble(at_configured, SIZE, OVERLAP, window_recorded=True).exact is True
    )


# ---------------------------------------------------------------------------
# The regression guard for the audited bug
# ---------------------------------------------------------------------------


def pre_fix_reassembly(
    chunks: list[tuple[int, str]], chunk_size: int, chunk_overlap: int
) -> tuple[str | None, bool]:
    """The algorithm as it stood before invariants 0 and 3a existed.

    Reimplemented here, in the test file, for one purpose: to prove the
    mismatched windows below are genuinely dangerous rather than hypothetical.
    The implication asserted in the regression test -- never `exact` AND wrong
    -- is vacuously true for any window that happens to reassemble correctly,
    so each case has to be shown to be one the old code got wrong.

    Returns (content, claimed_exact).
    """
    step = chunk_size - chunk_overlap
    contents = [content for _, content in chunks]
    out = contents[0]
    for k in range(1, len(contents)):
        shared = len(contents[k - 1]) - step
        if (
            shared < 0
            or shared > chunk_overlap
            or contents[k - 1][step:] != contents[k][:shared]
        ):
            return None, False
        out += contents[k][shared:]
    return out, len(contents[-1]) < chunk_size


# Every one of these reads a corpus cut at the configured SIZE/OVERLAP under a
# window whose step equals the stored chunk length, which is what drives `o` to
# 0. (SIZE, 0) is reachable from configuration alone -- app/config.py allows
# any 0 <= overlap < size -- and is the window the audit measured.
MISMATCHED_WINDOWS = [(SIZE, 0), (1200, 200), (2000, 1000)]


@pytest.mark.parametrize("read_size, read_overlap", MISMATCHED_WINDOWS)
@pytest.mark.parametrize("length", [1900, 2600, 3400, 5000])
def test_a_mismatched_window_never_claims_exact_for_the_wrong_text(
    length, read_size, read_overlap
):
    """The bug this change exists to kill, pinned as an implication.

    A 2,600-character document ingested at 1000/200 and read back at
    CHUNK_OVERLAP=0 returned 3,200 characters with `exact: true`. Stated as
    "not (exact and wrong)" rather than as a specific status code or reason,
    so the guard survives a reworded refusal, a different invariant catching
    it first, or a future version that legitimately recovers the text.
    """
    text = document_of_length(length)
    stored = store_as_ingest_would(text)  # cut at the configured window

    # The case is genuinely dangerous: the pre-fix algorithm claimed exactness
    # here and returned something other than the document. Without this, the
    # implication below could pass on a window that was never a hazard.
    pre_fix_content, pre_fix_claimed_exact = pre_fix_reassembly(
        stored, read_size, read_overlap
    )
    assert pre_fix_claimed_exact is True
    assert pre_fix_content != text

    result = reassemble(stored, read_size, read_overlap, window_recorded=False)

    assert not (result.exact and result.content != text), (
        f"length {length} read at {read_size}/{read_overlap}: claimed exact "
        f"and returned {len(result.content or '')} characters that are not the "
        f"document's {len(text)}"
    )


@pytest.mark.parametrize("read_size, read_overlap", MISMATCHED_WINDOWS)
def test_a_recorded_window_makes_the_mismatch_unreachable(read_size, read_overlap):
    """The structural half of the fix, not just the detection half.

    Detection is a safety net. The actual repair is that a document's window is
    recorded on its rows, so read-back never consults the mismatched window in
    the first place -- the document reassembles exactly however the service is
    configured afterwards. This asserts the net is a net, not the floor.
    """
    text = document_of_length(2600)
    stored = store_as_ingest_would(text)

    # What read-back does now: the recorded window, not the current config.
    recorded = reassemble(stored, SIZE, OVERLAP, window_recorded=True)
    assert recorded.exact is True
    assert recorded.content == text

    # And the mismatched window, had it been consulted, is still refused.
    assert (
        reassemble(stored, read_size, read_overlap, window_recorded=False).exact
        is False
    )


# ---------------------------------------------------------------------------
# join_chunk_run on sub-ranges: the core context expansion actually calls
#
# `reassemble` above only ever joins a whole document, so its runs always start
# at index 0 and always end at the document's tail. A context region is a
# SUB-RANGE: it legitimately starts at index 4 and legitimately ends in the
# middle of the note. Two things therefore have to hold that the whole-document
# tests above cannot see -- contiguity is checked relative to the run's own
# first index rather than against 0, and tail completeness applies only when
# the run really is the document's tail.
# ---------------------------------------------------------------------------


def all_chunks(text: str) -> list[tuple[int, str]]:
    """Every window chunk_text emits, unfiltered, with its index.

    Unfiltered on purpose: these tests are about the join, and the lengths used
    below are chosen so the 50-character minimum would drop nothing anyway. A
    dropped chunk is a hole, which is a different property with its own tests.
    """
    return list(enumerate(chunk_text(text, SIZE, OVERLAP)))


def expected_span(text: str, first: int, last: int) -> str:
    """The text a run of chunks `first..last` covers.

    Chunk j is `text[j*STEP : j*STEP + SIZE]`, so the run spans from the start
    of chunk `first` to the end of chunk `last`, clipped at the document end.
    This is the span definition the whole feature is judged against -- stated
    once, here, rather than recomputed per test.
    """
    return text[first * STEP : min(last * STEP + SIZE, len(text))]


# Lengths whose final chunk is comfortably above the 50-character minimum, so
# `all_chunks` is also what ingest would have stored.
_SUBRANGE_LENGTHS = [1200, 2500, 4000, 5000, 6000]


def _subrange_cases() -> list[tuple[int, int, int]]:
    cases = []
    for length in _SUBRANGE_LENGTHS:
        count = len(chunk_text(document_of_length(length), SIZE, OVERLAP))
        for first in range(count):
            for last in range(first, count):
                cases.append((length, first, last))
    return cases


@pytest.mark.parametrize("length, first, last", _subrange_cases())
def test_joining_a_sub_range_reproduces_exactly_the_span_it_covers(
    length, first, last
):
    """Every valid run of every one of these documents, joined exactly.

    This is context expansion's central correctness claim: the region handed to
    a caller is the real surrounding text, not an approximation of it and not
    the overlap duplicated at each seam. Asserted as string equality against an
    independently computed slice -- a length check or a `startswith` would pass
    against a join that duplicates the shared region.
    """
    text = document_of_length(length)
    chunks = all_chunks(text)
    is_tail = last == len(chunks) - 1

    result = join_chunk_run(
        chunks[first : last + 1],
        SIZE,
        OVERLAP,
        window_recorded=True,
        is_document_tail=is_tail,
    )

    assert result.exact is True, (
        f"length {length} run [{first}..{last}] refused with {result.reason!r}"
    )
    assert result.reason is None
    assert result.content == expected_span(text, first, last)


# Lengths from _SUBRANGE_LENGTHS with at least four chunks, so an interior run
# and a hole inside one both exist. Filtered here rather than skipped inside the
# test: a parametrization that cannot apply is not a case worth reporting.
_INTERIOR_RUN_LENGTHS = [
    length
    for length in _SUBRANGE_LENGTHS
    if len(chunk_text(document_of_length(length), SIZE, OVERLAP)) >= 4
]


@pytest.mark.parametrize("length", _INTERIOR_RUN_LENGTHS)
def test_a_sub_range_is_contiguous_relative_to_its_own_first_index(length):
    """A run starting at index 4 is not a run with four chunks missing.

    `reassemble` requires indices 0..n-1 because a whole document starts at 0.
    `join_chunk_run` must not: a mid-document region's first index is whatever
    the region starts at, and checking it against 0 would refuse every region
    except those at the start of a note.
    """
    text = document_of_length(length)
    chunks = all_chunks(text)

    interior = chunks[1:3]
    assert interior[0][0] == 1, "the run deliberately does not start at index 0"

    result = join_chunk_run(
        interior,
        SIZE,
        OVERLAP,
        window_recorded=True,
        is_document_tail=False,
    )

    assert result.exact is True, f"refused with {result.reason!r}"
    assert result.content == expected_span(text, 1, 2)

    # A genuine hole inside the run is still refused -- relative contiguity is
    # a different origin, not a weaker check.
    with_hole = [chunks[1], chunks[3]]
    holed = join_chunk_run(
        with_hole, SIZE, OVERLAP, window_recorded=True, is_document_tail=False
    )
    assert holed.exact is False
    assert holed.content is None
    assert "not contiguous" in holed.reason


def _full_size_interior_run(length: int) -> tuple[str, list[tuple[int, str]], int, int]:
    """A run whose LAST chunk is full size and which is not the document tail.

    Tail completeness only has something to say about a run ending on a
    full-size chunk, so a test of the flag has to find one. `chunk_text` never
    emits a full-size FINAL chunk, so such a run necessarily ends mid-document
    -- which is exactly the shape context expansion produces all day.
    """
    text = document_of_length(length)
    chunks = all_chunks(text)
    for last in range(len(chunks) - 2, 0, -1):
        if len(chunks[last][1]) == SIZE:
            return text, chunks, 1, last
    raise AssertionError(f"length {length} has no full-size interior chunk")


@pytest.mark.parametrize("length", [4000, 5000, 6000])
def test_is_document_tail_is_what_decides_a_full_size_final_chunk(length):
    """The same run, both ways -- exact with False, refused with True.

    Asserting both halves is the point. A test that only checked
    `is_document_tail=False` would keep passing if tail completeness had gone
    dead altogether, and a test that only checked `True` would keep passing if
    the flag were ignored and the check always applied. Pinning that the flag
    is what makes the difference is what proves it is wired up.

    The behaviour itself: a run ending on a full-size chunk means a successor
    window existed. At the end of a document that successor was dropped at
    ingest and its text is gone, which invariant 5 reports. Mid-document the
    successor is stored and merely outside the region, which is not a loss --
    and reporting it as one would mark every neighbours region inexact.
    """
    text, chunks, first, last = _full_size_interior_run(length)
    run = chunks[first : last + 1]
    assert len(run[-1][1]) == SIZE, "the run must end on a full-size chunk"
    assert last < len(chunks) - 1, "the run must not be the document's tail"

    interior = join_chunk_run(
        run, SIZE, OVERLAP, window_recorded=True, is_document_tail=False
    )
    assert interior.exact is True, f"refused with {interior.reason!r}"
    assert interior.content == expected_span(text, first, last)

    as_tail = join_chunk_run(
        run, SIZE, OVERLAP, window_recorded=True, is_document_tail=True
    )
    assert as_tail.exact is False
    assert "a successor chunk existed and was dropped" in as_tail.reason
    # The text is the same either way -- only the verdict differs, because the
    # question the flag answers is "is anything missing after this?", not
    # "what does this say?".
    assert as_tail.content == interior.content


@pytest.mark.parametrize("length", _SUBRANGE_LENGTHS)
def test_the_document_tail_run_is_still_judged_as_a_tail(length):
    """The flag's True branch on a real tail, where it is the correct answer.

    `chunk_text` never emits a full-size final chunk, so a genuine, complete
    document tail passes invariant 5 -- the refusal is reserved for a tail that
    was truncated by the 50-character minimum.
    """
    text = document_of_length(length)
    chunks = all_chunks(text)
    last = len(chunks) - 1
    assert len(chunks[last][1]) < SIZE

    result = join_chunk_run(
        chunks[last:], SIZE, OVERLAP, window_recorded=True, is_document_tail=True
    )

    assert result.exact is True, f"refused with {result.reason!r}"
    assert result.content == expected_span(text, last, last)


def test_reassemble_and_join_chunk_run_agree_on_a_whole_document():
    """The wrapper must not have drifted from the core it delegates to.

    `reassemble` is documented as `join_chunk_run` plus the two whole-document
    requirements. Asserting the two produce the same text and verdict on a
    whole document is what keeps that true.
    """
    text = document_of_length(4000)
    chunks = all_chunks(text)

    whole = reassemble(chunks, SIZE, OVERLAP, window_recorded=True)
    run = join_chunk_run(
        chunks, SIZE, OVERLAP, window_recorded=True, is_document_tail=True
    )

    assert (whole.content, whole.exact, whole.reason) == (
        run.content,
        run.exact,
        run.reason,
    )
    assert whole.content == text
