"""Loading a corpus of markdown notes with YAML frontmatter.

A note is the unit the consumer cares about: retrieval is judged on whether the
right *note* came back, whichever strategy indexed it and at whatever
granularity.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


class CorpusError(RuntimeError):
    """The corpus directory could not be read as a corpus."""


@dataclass(frozen=True)
class Note:
    """One markdown note.

    note_id is the path relative to the corpus root, forward-slashed. It is what
    a query set refers to, so it has to be stable and readable by a person
    editing that file by hand -- a content hash would be neither.
    """

    note_id: str
    name: str
    description: str
    note_type: str
    body: str
    # True when this note's frontmatter was not valid YAML and the line-based
    # fallback in split_frontmatter had to recover the fields. Counted and
    # reported, never silent: see that function's docstring for why the count
    # matters to the comparison.
    frontmatter_recovered: bool = False

    @property
    def has_description(self) -> bool:
        return bool(self.description.strip())


# A hand-written frontmatter line: a bare key, a colon, then the rest of the
# line verbatim. Deliberately NOT a YAML subset -- the whole point is to read
# lines a YAML parser rejects.
_FRONTMATTER_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)$")


def recover_frontmatter(raw: str) -> dict:
    """Read hand-written `key: value` lines that yaml.safe_load refused.

    WHY THIS EXISTS, because a tolerant parser is normally the wrong answer:
    21% of the real brain's notes (93 of 440 at the time of writing) have
    frontmatter that a YAML parser rejects, almost always for one of two
    reasons -- an unquoted `:` inside a plain scalar ("description: isolation:
    worktree checked ..."), or a value opening with `{` or a backtick, which
    YAML reads as a flow mapping or an unknown indicator. In every sampled case
    the human DID write a real, substantial description; only the quoting is
    wrong.

    Treating those 93 notes as having no description is not a neutral choice.
    It would strip the `summary` strategy of its retrieval key on a fifth of
    the corpus and hand `chunk` a ~21-point advantage that has nothing to do
    with indexing granularity and everything to do with a fixable quoting bug
    in the source notes. Measuring that as a property of the strategy would be
    a confident wrong answer, which is the one outcome this harness exists to
    avoid.

    So the fields are recovered, and every report states how many notes needed
    it, so a reader can discount the result if the count is high.

    Only the first occurrence of a key wins, and only top-level single-line
    scalars are read: nested mappings, block scalars and lists are out of
    reach. That is sufficient for `name`, `description` and `type`, which are
    the only three fields this harness uses.
    """
    recovered: dict[str, str] = {}
    for line in raw.splitlines():
        match = _FRONTMATTER_LINE.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key in recovered:
            continue
        # Strip one layer of matched quotes, which a hand-writer sometimes adds
        # and which YAML would have consumed.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        recovered[key] = value
    return recovered


# How to read a frontmatter block. Exposed as `--frontmatter` on the CLI rather
# than decided here, because the choice MOVES THE RESULT and no amount of
# argument settles it as well as running it both ways does.
#
#   strict  - yaml.safe_load only. A block that fails to parse contributes
#             nothing, and a value YAML silently eats as a `#` comment stays
#             eaten. This is what a production ingest using a stock YAML
#             parser would see.
#   recover - additionally read the authored text for keys YAML dropped or
#             truncated. This is what the human wrote.
FRONTMATTER_MODES = ("recover", "strict")


def _merge_recovered(metadata: dict, raw: str) -> tuple[dict, bool]:
    """Restore authored characters that a successful YAML parse dropped.

    A block can parse cleanly and still lose most of a description, because an
    unescaped `#` after whitespace starts a YAML comment: `description: fixed
    in #562; also ...` parses to "fixed in" and the rest is silently gone. 23
    notes in the real brain are truncated this way and 2 are nulled outright,
    on top of the 93 whose block does not parse at all.

    THE MERGE RULE, and why it is safe in both directions: take the line-based
    value only when it is strictly LONGER than the parsed one. The line reader
    sees the authored characters verbatim, so YAML can only differ from it by
    shortening (a comment strip) or by lengthening (joining a block scalar or a
    multi-line quoted value, which the line reader cannot see at all). Longer
    line value therefore means exactly "YAML dropped authored text", and
    longer YAML value means exactly "the author used a construct worth
    keeping". Neither case can damage the other.
    """
    recovered = recover_frontmatter(raw)
    merged = dict(metadata)
    changed = False
    for key, authored in recovered.items():
        existing = metadata.get(key)
        existing_text = "" if existing is None else str(existing)
        if len(authored) > len(existing_text):
            merged[key] = authored
            changed = True
    return merged, changed


def split_frontmatter(
    text: str, mode: str = "recover"
) -> tuple[dict, str, bool]:
    """Split a `---` delimited YAML frontmatter block off the front of `text`.

    Returns (metadata, body, recovered). A note with no frontmatter yields
    ({}, text, False) rather than an error: the real brain is hand-maintained
    and this harness should report a note it cannot summary-index, not refuse
    to load the corpus because of one.
    """
    if mode not in FRONTMATTER_MODES:
        raise CorpusError(
            f"unknown frontmatter mode {mode!r}; expected one of "
            f"{', '.join(FRONTMATTER_MODES)}"
        )
    if not text.startswith("---"):
        return {}, text, False
    # Split on the *closing* delimiter only. A body containing a `---` thematic
    # break must not be mistaken for the end of the frontmatter, so the search
    # starts after the opening line.
    lines = text.splitlines()
    for position in range(1, len(lines)):
        if lines[position].strip() == "---":
            raw = "\n".join(lines[1:position])
            body = "\n".join(lines[position + 1 :])
            try:
                metadata = yaml.safe_load(raw) or {}
            except yaml.YAMLError:
                if mode == "strict":
                    return {}, body, False
                return recover_frontmatter(raw), body, True
            if not isinstance(metadata, dict):
                if mode == "strict":
                    return {}, body, False
                return recover_frontmatter(raw), body, True
            if mode == "strict":
                return metadata, body, False
            merged, changed = _merge_recovered(metadata, raw)
            return merged, body, changed
    # An unterminated opening delimiter: treat the whole file as body.
    return {}, text, False


def _as_text(value: object) -> str:
    """Frontmatter is hand-written, so a field may be absent, null or a number."""
    if value is None:
        return ""
    return str(value).strip()


def load_note(path: Path, root: Path, mode: str = "recover") -> Note:
    text = path.read_text(encoding="utf-8")
    metadata, body, recovered = split_frontmatter(text, mode)
    note_id = path.relative_to(root).as_posix()
    # `name` is this corpus's convention; `title` is accepted because it is the
    # other common frontmatter spelling and a corpus using it should measure
    # rather than silently score zero on the summary strategy. The filename stem
    # is the last resort so a note always has some human-written handle.
    name = (
        _as_text(metadata.get("name"))
        or _as_text(metadata.get("title"))
        or path.stem.replace("-", " ").replace("_", " ")
    )
    return Note(
        note_id=note_id,
        name=name,
        description=_as_text(metadata.get("description")),
        note_type=_as_text(metadata.get("type")),
        body=body.strip(),
        frontmatter_recovered=recovered,
    )


def load_corpus(root: Path, mode: str = "recover") -> list[Note]:
    """Load every `*.md` under `root`, recursively, sorted by note_id."""
    if not root.is_dir():
        raise CorpusError(f"corpus path is not a directory: {root}")
    notes = [load_note(path, root, mode) for path in sorted(root.rglob("*.md"))]
    if not notes:
        raise CorpusError(f"no *.md files found under {root}")
    return sorted(notes, key=lambda note: note.note_id)


def corpus_shape(notes: list[Note]) -> dict:
    """Descriptive stats, so a report says what it was actually run against.

    The claim under test is argued from corpus shape, so every run records the
    shape it saw. A run whose median body length or description coverage looks
    nothing like the real brain is not evidence about the real brain.
    """

    def median(values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2

    headings = [
        sum(1 for line in note.body.splitlines() if line.startswith("#"))
        for note in notes
    ]
    return {
        "notes": len(notes),
        "median_body_chars": median([len(note.body) for note in notes]),
        "median_name_chars": median([len(note.name) for note in notes]),
        "median_description_chars": median(
            [len(note.description) for note in notes if note.has_description]
        ),
        "notes_with_description": sum(1 for note in notes if note.has_description),
        "notes_with_type": sum(1 for note in notes if note.note_type),
        "median_markdown_headings": median(headings),
        # How much of the description coverage above rests on the line-based
        # fallback rather than on valid YAML. A high number does not invalidate
        # the run, but it does say the source corpus needs its frontmatter
        # quoted before any production ingest can rely on these fields.
        "notes_with_recovered_frontmatter": sum(
            1 for note in notes if note.frontmatter_recovered
        ),
    }
