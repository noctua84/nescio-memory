"""Pure Markdown splitter: one ingested document becomes one or more passages.

This module is deliberately free of configuration and persistence. It imports
neither ``app.config`` nor any model or repository, so it is testable without a
database and without environment variables; ``app/core/pipeline.py`` owns
everything that needs settings.
"""

import re
from dataclasses import dataclass
from datetime import date

import yaml

# Any change to the algorithm below that can move the passage boundaries of an
# already-ingested note MUST bump this constant. It is part of the ingest skip
# key, so a note whose boundaries would change is only re-split when the
# version it was ingested under no longer matches.
#
# It starts at "2", not "1", because the split threshold changed during
# planning: structure is now used only when the body *exceeds* TARGET_CHARS.
# Any corpus loaded against a prototype of the earlier rule must therefore
# re-split rather than be skipped.
#
# "3": short-section runs no longer collapse under one heading.
SPLITTER_VERSION = "3"

# TARGET_CHARS and MAX_CHARS are two different things, not one threshold used
# twice. TARGET_CHARS decides *whether the document's structure is worth
# using* (and is the soft target when packing paragraphs); MAX_CHARS is the
# hard cap on a single passage.
#
# The gap between them is measured, not guessed: of the 458 notes in the ai-os
# corpus at or under 4 KB, 64 carry three or more `##` headings and all 64 sit
# in the 1.5-4 KB band. A 4 KB "small note stays whole" threshold would have
# left every one of those 64 genuinely structured notes as a single
# multi-topic passage -- exactly the "here is a page, read it again" failure
# this model exists to remove. A headingless 3 KB note, by contrast, has no
# structure to exploit and stays whole.
TARGET_CHARS = 1_500
MAX_CHARS = 4_000

# A section shorter than this (stripped) is merged into a neighbour rather
# than becoming a passage of its own: a two-line "Status" section is not an
# independently retrievable answer to anything.
MIN_SECTION_CHARS = 300

# There is no *minimum passage length*. The MIN_CONTENT_CHARS = 50 floor that
# /api/v1 applies is deliberately absent here: it silently dropped short
# notes, and a short note is still worth finding.

# `#` (H1) is not a split point. A document opening with `# ADR-0003: ...`
# followed by a status block keeps that text as the preamble.
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$")
_FENCE_PREFIXES = ("```", "~~~")

# U+203A with a space either side, e.g.
# "Decision › Eviction order" for a `### Eviction order` under `## Decision`.
_HEADING_SEP = " › "

# The provenance line ai-os writes (see
# ~/PycharmProjects/ai-os/scripts/promote_learnings.py). The separator is an
# em-dash *or* a hyphen, hence the alternation.
_PROVENANCE_RE = re.compile(
    r"\[Source:\s*(user override|empirical|agent inference)\s*[—-]\s*"
    r"(\d{4}-\d{2}-\d{2})\s*\]"
)

# Blank-line paragraph boundary. A "blank" line may carry spaces or tabs.
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n+")

_FRONTMATTER_FIELDS = ("name", "description", "type")


@dataclass(frozen=True)
class Passage:
    ordinal: int
    heading_path: str | None
    content: str


@dataclass(frozen=True)
class SplitResult:
    name: str | None
    description: str | None
    type: str | None
    provenance_kind: str | None
    provenance_date: date | None
    body: str
    passages: list[Passage]


@dataclass(frozen=True)
class _Section:
    """A heading-delimited span of the body, including its own heading line."""

    heading_path: str | None
    text: str


@dataclass
class _Unit:
    """A section (or merged run of sections) before oversize packing."""

    heading_path: str | None
    text: str


def _strip_frontmatter(content: str) -> tuple[dict[str, str | None], str]:
    """Split YAML frontmatter from the body.

    Returns the three typed frontmatter fields and the remaining body. A
    malformed block must never raise: a YAML error, a non-mapping document or
    a missing closing `---` all mean "this document has no frontmatter", and
    the delimiter lines stay in the body so nothing is silently lost.
    """
    fields: dict[str, str | None] = dict.fromkeys(_FRONTMATTER_FIELDS)
    lines = content.split("\n")
    if not lines or lines[0].rstrip() != "---":
        return fields, content

    closing = next(
        (i for i in range(1, len(lines)) if lines[i].rstrip() == "---"), None
    )
    if closing is None:
        return fields, content

    try:
        parsed = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError:
        return fields, content
    if not isinstance(parsed, dict):
        return fields, content

    for key in _FRONTMATTER_FIELDS:
        value = parsed.get(key)
        # Only a genuine string counts. A date, a list or a nested mapping is
        # not a name, and coercing one with str() would store "2026-10-29" or
        # "['a', 'b']" as if the author had written it.
        fields[key] = value if isinstance(value, str) else None
    return fields, "\n".join(lines[closing + 1 :])


def _find_provenance(body: str) -> tuple[str | None, date | None]:
    """First provenance marker anywhere in the body, or two ``None``s.

    The line is *not* removed from the body and is not stripped from the
    passage containing it: it is part of what the author wrote.
    """
    match = _PROVENANCE_RE.search(body)
    if match is None:
        return None, None
    try:
        parsed = date.fromisoformat(match.group(2))
    except ValueError:
        # A syntactically well-formed but impossible date (2026-13-45) makes
        # the whole marker untrustworthy, so neither half is kept.
        return None, None
    return match.group(1), parsed


def _scan_sections(body: str) -> list[_Section]:
    """Cut the body at `##`/`###` headings that are outside fenced code.

    Always returns at least one section. Text before the first heading is the
    preamble, with ``heading_path=None``. Each section includes its own
    heading line so that a passage reads naturally on its own.
    """
    sections: list[_Section] = []
    current: list[str] = []
    heading_path: str | None = None
    last_h2: str | None = None
    in_fence = False

    def flush() -> None:
        text = "\n".join(current).rstrip()
        # An empty preamble is not a section; an empty body still yields one
        # (see the final fallback below).
        if text or heading_path is not None:
            sections.append(_Section(heading_path=heading_path, text=text))

    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith(_FENCE_PREFIXES):
            in_fence = not in_fence
            current.append(line)
            continue
        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            current.append(line)
            continue

        flush()
        current = [line]
        title = match.group(2).strip()
        if len(match.group(1)) == 2:
            last_h2 = title
            heading_path = title
        else:
            # A `###` with no preceding `##` gets its own title alone.
            heading_path = f"{last_h2}{_HEADING_SEP}{title}" if last_h2 else title

    flush()
    if not sections:
        sections.append(_Section(heading_path=None, text=body.rstrip()))
    return sections


def _merge_short_sections(sections: list[_Section]) -> list[_Unit]:
    """Fold sections below ``MIN_SECTION_CHARS`` into a neighbour.

    A short section merges *backward* into the previous unit while the
    result stays at or under TARGET_CHARS (unchanged).

    A short section with no previous unit yet -- the document opens with
    one or more short sections -- instead merges *forward* into ``pending``,
    which takes the heading path of whatever eventually absorbs it: a long
    section it joins, or its own first entry if the whole run stays short.

    The forward-merge has a stop rule: once adding the next short section to
    ``pending`` would push the group over TARGET_CHARS, ``pending`` is
    flushed as its own unit -- labelled with its *first* section's heading
    path -- before that section is considered at all. The section is then
    re-evaluated against the freshly flushed unit under the ordinary
    backward-merge rule above (merge in if it still fits, else stand alone).
    Without this stop, a long run of short sections (e.g. hundreds of daily
    log entries) would collapse into one oversized unit carrying only the
    first entry's heading path.
    """
    units: list[_Unit] = []
    pending: list[_Section] = []

    for section in sections:
        short = len(section.text.strip()) < MIN_SECTION_CHARS

        if short and pending:
            candidate = "\n\n".join(p.text for p in pending) + "\n\n" + section.text
            if len(candidate) <= TARGET_CHARS:
                pending.append(section)
                continue
            # The pending run can't absorb another section without outgrowing
            # TARGET_CHARS. Flush it now, labelled with its first heading, and
            # fall through to re-check `section` against the unit just created.
            units.append(
                _Unit(
                    heading_path=pending[0].heading_path,
                    text="\n\n".join(p.text for p in pending),
                )
            )
            pending = []

        if short and units:
            previous = units[-1]
            candidate = f"{previous.text}\n\n{section.text}"
            # Stop merging into a unit once it would outgrow TARGET_CHARS:
            # past that point the short section is better off alone than
            # making an already-large passage larger and vaguer.
            if len(candidate) <= TARGET_CHARS:
                previous.text = candidate
                continue
        elif short:
            # The first section has no previous unit to merge back into, so
            # it merges *forward* and the merged unit takes the heading path
            # of the section it joins.
            pending.append(section)
            continue
        units.append(
            _Unit(
                heading_path=section.heading_path,
                text="\n\n".join([p.text for p in pending] + [section.text]),
            )
        )
        pending = []

    if pending:
        # Every remaining section was short. There is no "next" to inherit
        # from, so the merged unit keeps the first section's heading path.
        units.append(
            _Unit(
                heading_path=pending[0].heading_path,
                text="\n\n".join(p.text for p in pending),
            )
        )
    return units


def _hard_split(paragraph: str) -> list[str]:
    """Last resort for a single paragraph longer than ``MAX_CHARS``."""
    if len(paragraph) <= MAX_CHARS:
        return [paragraph]
    return [paragraph[i : i + MAX_CHARS] for i in range(0, len(paragraph), MAX_CHARS)]


def _pack(text: str) -> list[str]:
    """Break a unit over ``MAX_CHARS`` on paragraph boundaries, no overlap.

    A unit at or under the cap passes through unchanged, which is why a small
    note and a headingless 112 KB PDF extraction share one code path.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    pieces: list[str] = []
    buffer = ""
    for paragraph in _PARAGRAPH_RE.split(text):
        if not paragraph:
            continue
        for chunk in _hard_split(paragraph):
            if not buffer:
                buffer = chunk
            elif len(buffer) + 2 + len(chunk) > TARGET_CHARS:
                pieces.append(buffer)
                buffer = chunk
            else:
                buffer = f"{buffer}\n\n{chunk}"
    if buffer or not pieces:
        pieces.append(buffer)
    return pieces


def split_note(content: str) -> SplitResult:
    """Split one Markdown document into its typed fields, body and passages.

    Ordinals are 0-based and contiguous, every character of ``body`` appears
    in exactly one passage (whitespace at the joins aside), no passage exceeds
    ``MAX_CHARS``, and there is always at least one passage -- an empty body
    yields a single passage with ``content=""`` so the note stays addressable.
    """
    fields, raw_body = _strip_frontmatter(content)

    # Leading blank *lines* are an artefact of removing the frontmatter block
    # and trailing whitespace is noise, but indentation on the first content
    # line is not -- it may open a code block -- so only whole blank lines go.
    body_lines = raw_body.split("\n")
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body = "\n".join(body_lines).rstrip()

    provenance_kind, provenance_date = _find_provenance(body)

    sections = _scan_sections(body)
    has_heading = any(section.heading_path is not None for section in sections)
    if has_heading and len(body) > TARGET_CHARS:
        units = _merge_short_sections(sections)
    else:
        # No usable structure, or small enough that structure is not worth
        # using. The whole body is one unit; packing below still enforces
        # MAX_CHARS, which is what turns a headingless 112 KB document into
        # several passages without a second code path.
        units = [_Unit(heading_path=None, text=body)]

    passages: list[Passage] = []
    for unit in units:
        for piece in _pack(unit.text):
            passages.append(
                Passage(
                    ordinal=len(passages),
                    heading_path=unit.heading_path,
                    content=piece,
                )
            )

    return SplitResult(
        name=fields["name"],
        description=fields["description"],
        type=fields["type"],
        provenance_kind=provenance_kind,
        provenance_date=provenance_date,
        body=body,
        passages=passages,
    )
