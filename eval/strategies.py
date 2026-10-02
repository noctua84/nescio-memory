"""The two indexing strategies under comparison, behind one interface.

A strategy is nothing more than a rule for turning one note into the list of
texts that get embedded, each of which becomes one retrievable unit pointing
back at that note. Both strategies then go through the identical embedder,
index builder and scorer, so the only variable between two runs is this rule.
"""
from dataclasses import dataclass
from typing import Callable

import eval.appenv  # noqa: F401  -- must precede any app. import; see appenv

from app.config import settings
from app.core.chunking import chunk_text
from eval.corpus import Note

# Imported, not copied, on purpose. The ingest endpoint drops any chunk with
# fewer than this many non-whitespace characters, so a `chunk` strategy that
# kept them would be measuring a pipeline the service does not run. A mirrored
# literal here could drift from the endpoint silently; an import cannot.
from app.api.v1.ingest import MIN_CONTENT_CHARS


@dataclass(frozen=True)
class Strategy:
    name: str
    summary: str
    unit_texts: Callable[[Note], list[str]]

    def describe(self) -> str:
        return self.summary


def _chunk_units(note: Note) -> list[str]:
    """The service's current behaviour, as app/api/v1/ingest.py performs it.

    chunk_text reads CHUNK_SIZE/CHUNK_OVERLAP from settings, so this follows the
    deployed configuration rather than hardcoding 1000/200 -- the run records
    the values it actually used.
    """
    return [
        chunk for chunk in chunk_text(note.body)
        if len(chunk.strip()) >= MIN_CONTENT_CHARS
    ]


def _summary_units(note: Note) -> list[str]:
    """One unit per note: the human-written name and description.

    Notes with no description fall back to the name alone rather than being
    dropped. Dropping them would give the two strategies different candidate
    pools and make the comparison meaningless -- a note the summary strategy
    never indexed cannot be retrieved by it, which would read as a recall loss
    caused by the strategy rather than by missing metadata. The count of notes
    that needed this fallback is reported, because at a high enough count the
    strategy is being flattered by the fallback.
    """
    if note.has_description:
        return [f"{note.name}\n{note.description}"]
    return [note.name] if note.name.strip() else []


CHUNK = Strategy(
    name="chunk",
    summary=(
        f"chunk_text(body) at CHUNK_SIZE={settings.chunk_size}/"
        f"CHUNK_OVERLAP={settings.chunk_overlap}, dropping chunks under "
        f"{MIN_CONTENT_CHARS} stripped chars, as app/api/v1/ingest.py does"
    ),
    unit_texts=_chunk_units,
)

SUMMARY = Strategy(
    name="summary",
    summary="one unit per note: name + '\\n' + description (ADR 0005's proposal)",
    unit_texts=_summary_units,
)

STRATEGIES = {strategy.name: strategy for strategy in (CHUNK, SUMMARY)}
