"""Loading a corpus of markdown notes with YAML frontmatter.

A note is the unit the consumer cares about: retrieval is judged on whether the
right *note* came back, whichever strategy indexed it and at whatever
granularity.
"""
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

    @property
    def has_description(self) -> bool:
        return bool(self.description.strip())


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---` delimited YAML frontmatter block off the front of `text`.

    Returns (metadata, body). A note with no frontmatter yields ({}, text)
    rather than an error: the real brain is hand-maintained and this harness
    should report a note it cannot summary-index, not refuse to load the corpus
    because of one.
    """
    if not text.startswith("---"):
        return {}, text
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
            except yaml.YAMLError as exc:
                raise CorpusError(f"unparseable frontmatter: {exc}") from exc
            if not isinstance(metadata, dict):
                raise CorpusError(
                    f"frontmatter was {type(metadata).__name__}, expected a mapping"
                )
            return metadata, body
    # An unterminated opening delimiter: treat the whole file as body.
    return {}, text


def _as_text(value: object) -> str:
    """Frontmatter is hand-written, so a field may be absent, null or a number."""
    if value is None:
        return ""
    return str(value).strip()


def load_note(path: Path, root: Path) -> Note:
    text = path.read_text(encoding="utf-8")
    metadata, body = split_frontmatter(text)
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
    )


def load_corpus(root: Path) -> list[Note]:
    """Load every `*.md` under `root`, recursively, sorted by note_id."""
    if not root.is_dir():
        raise CorpusError(f"corpus path is not a directory: {root}")
    notes = [load_note(path, root) for path in sorted(root.rglob("*.md"))]
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
    }
