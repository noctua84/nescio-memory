"""Building an in-memory index and scanning it, with no database.

Deliberately not pgvector. Holding the vectors here isolates the measurement to
the indexing strategy: an exact brute-force scan has perfect recall of its own,
so any difference between two runs is the strategy and not the index. The cost
is that this says nothing about what production's HNSW index loses -- that
limitation is restated in the generated report, not just here.
"""
import sys
from dataclasses import dataclass, field

from eval.corpus import Note
from eval.strategies import Strategy
from eval.vectors import dot, l2_normalize


@dataclass(frozen=True)
class Unit:
    """One retrievable thing: a chunk, or a whole note's summary.

    `note_id` is the payload that matters -- it is what identifies the parent
    note, and what the query set is written in terms of.
    """

    note_id: str
    unit_index: int
    chars: int
    vector: list[float]


@dataclass
class Index:
    strategy_name: str
    units: list[Unit] = field(default_factory=list)
    # Notes the strategy produced no unit for. They are unreachable by this
    # strategy, so a report that did not name them would show an unexplained
    # recall ceiling.
    notes_without_units: list[str] = field(default_factory=list)
    # Notes whose summary had to fall back to the name alone.
    notes_without_description: list[str] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len({unit.note_id for unit in self.units})

    def stats(self) -> dict:
        return {
            "units": len(self.units),
            "notes_indexed": self.note_count,
            "units_per_note": (
                round(len(self.units) / self.note_count, 2)
                if self.note_count
                else 0.0
            ),
            "mean_unit_chars": (
                round(sum(unit.chars for unit in self.units) / len(self.units), 1)
                if self.units
                else 0.0
            ),
            "notes_without_units": self.notes_without_units,
            "notes_without_description": len(self.notes_without_description),
        }


def build_index(notes: list[Note], strategy: Strategy, embedder) -> Index:
    index = Index(strategy_name=strategy.name)
    print(
        f"[{strategy.name}] embedding units for {len(notes)} notes...",
        file=sys.stderr,
        flush=True,
    )
    for note in notes:
        texts = strategy.unit_texts(note)
        if not texts:
            index.notes_without_units.append(note.note_id)
            continue
        if strategy.name == "summary" and not note.has_description:
            index.notes_without_description.append(note.note_id)
        for position, text in enumerate(texts):
            index.units.append(
                Unit(
                    note_id=note.note_id,
                    unit_index=position,
                    chars=len(text),
                    # Normalized once here so every query is a plain dot product.
                    vector=l2_normalize(embedder(text)),
                )
            )
    return index


@dataclass(frozen=True)
class Hit:
    """A retrieved unit. `rank` is 1-based in UNIT space, as production ranks."""

    rank: int
    note_id: str
    similarity: float
    unit_index: int


def search(index: Index, query_vector: list[float], limit: int) -> list[Hit]:
    """The top `limit` units by cosine similarity, highest first.

    Ties are broken by (note_id, unit_index) so a run is reproducible for a
    fixed embedder rather than depending on insertion order via a stable sort.
    """
    query = l2_normalize(query_vector)
    scored = [
        (dot(unit.vector, query), unit.note_id, unit.unit_index, unit)
        for unit in index.units
    ]
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [
        Hit(
            rank=position + 1,
            note_id=unit.note_id,
            similarity=similarity,
            unit_index=unit.unit_index,
        )
        for position, (similarity, _, _, unit) in enumerate(scored[:limit])
    ]
