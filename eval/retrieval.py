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

    `unit_kind` is the name of the strategy that produced the unit. For a
    single-strategy index it is redundant; for the `hybrid` index it is the
    only thing that tells the two unit types apart, which matters both for a
    deterministic tie-break (a chunk and a summary of the same note share a
    unit_index of 0) and for reporting which route reached a note.
    """

    note_id: str
    unit_kind: str
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
        by_kind: dict[str, int] = {}
        for unit in self.units:
            by_kind[unit.unit_kind] = by_kind.get(unit.unit_kind, 0) + 1
        return {
            "units": len(self.units),
            "units_by_kind": by_kind,
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
                    unit_kind=strategy.name,
                    unit_index=position,
                    chars=len(text),
                    # Normalized once here so every query is a plain dot product.
                    vector=l2_normalize(embedder(text)),
                )
            )
    return index


def union_index(name: str, parts: list[Index]) -> Index:
    """One index over the units of several others. Embeds nothing.

    This is how the `hybrid` strategy is built, and building it by composition
    rather than by a third pass over the corpus is a hard requirement, not a
    convenience. The embedder is called serially, one HTTP round trip per unit,
    so on the real corpus a third pass would be another ~2,800 calls and
    roughly another twenty minutes to recompute vectors that are already in
    memory and bit-identical. Composing the finished indexes makes re-embedding
    structurally impossible rather than merely unlikely -- a text cache would
    give the same answer only as long as nobody changed how a unit's text is
    derived.

    `notes_without_units` is the INTERSECTION across parts, not the union: a
    note is unreachable in the combined index only if no part could index it.
    Getting that backwards would report a recall ceiling the index does not
    have.
    """
    if not parts:
        raise ValueError("union_index needs at least one part")
    combined = Index(strategy_name=name)
    for part in parts:
        combined.units.extend(part.units)
    unreachable = set(parts[0].notes_without_units)
    for part in parts[1:]:
        unreachable &= set(part.notes_without_units)
    combined.notes_without_units = sorted(unreachable)
    # Informational and carried through from whichever part reports it: the
    # count says how many notes contributed a name-only summary unit, which is
    # still true of those units inside the union.
    seen: set[str] = set()
    for part in parts:
        for note_id in part.notes_without_description:
            if note_id not in seen:
                seen.add(note_id)
                combined.notes_without_description.append(note_id)
    return combined


@dataclass(frozen=True)
class Hit:
    """A retrieved unit. `rank` is 1-based in UNIT space, as production ranks."""

    rank: int
    note_id: str
    similarity: float
    unit_kind: str
    unit_index: int


def search(index: Index, query_vector: list[float], limit: int) -> list[Hit]:
    """The top `limit` units by cosine similarity, highest first.

    Ties are broken by (note_id, unit_kind, unit_index) so a run is
    reproducible for a fixed embedder rather than depending on insertion order
    via a stable sort. unit_kind is in the key because in the `hybrid` index a
    note's chunk 0 and its summary unit would otherwise be indistinguishable.

    ONE RANKING, NOT A FUSION. The hybrid index puts both unit types in a
    single cosine ordering rather than ranking each route separately and
    merging with something like RRF. That is the right call here for two
    reasons. First, the scores are directly comparable: both unit types come
    from the same embedding model under the same metric, so there is no scale
    mismatch for a fusion to correct -- fusion earns its keep when combining
    incommensurable scorers such as BM25 with a dense vector, and inventing a
    rank-based merge here would discard real score information in exchange for
    nothing. Second, this is the design that could actually ship: /search
    issues one ORDER BY over one table, so a single shared ranking is what
    "store both unit types" would mean in production. A fusion would be a
    different, larger proposal and should be measured as one if anyone wants
    it.
    """
    query = l2_normalize(query_vector)
    scored = [
        (dot(unit.vector, query), unit.note_id, unit.unit_kind, unit.unit_index, unit)
        for unit in index.units
    ]
    scored.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    return [
        Hit(
            rank=position + 1,
            note_id=unit.note_id,
            similarity=similarity,
            unit_kind=unit.unit_kind,
            unit_index=unit.unit_index,
        )
        for position, (similarity, _, _, _, unit) in enumerate(scored[:limit])
    ]
