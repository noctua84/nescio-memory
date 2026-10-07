"""Loading and validating a query set.

FORMAT: YAML, not JSON. Three reasons, in order of weight:

  1. A query set is prose that a human maintains by hand, and the most valuable
     field in a case is often the `why` -- the reason this query should find
     that note. JSON has no comments, so the reasoning around a group of cases
     has nowhere to live except inside string values.
  2. Query strings and rationales are natural-language sentences. YAML block
     scalars carry them without escaping; JSON makes every apostrophe and
     newline an editing hazard, which discourages adding cases.
  3. pyyaml is already a declared dependency of this project, so this costs
     nothing to add.

Machine-readable OUTPUT stays JSON (stdlib, see report.py). Different job:
nothing hand-edits it.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from eval.corpus import Note


class QuerySetError(RuntimeError):
    """The query set is unusable. Always fatal -- never degraded into a run."""


# The three case kinds, closed on purpose.
#
# `kind` is REQUIRED on every case and is checked against this tuple, because
# per-class reporting is the only thing that makes an aggregate interpretable.
# An aggregate is driven entirely by the case mix: shift the mix and the
# headline number moves without any retrieval behaviour changing, and a reader
# has no way to see it happen. Reporting recall and MRR per class instead makes
# the claim falsifiable -- "summary loses on oblique queries" is a statement
# someone can go and check.
#
# Free-form kinds are rejected rather than accepted, because a set that drifts
# into eight near-synonymous labels fragments into singleton classes and the
# per-class numbers stop meaning anything.
#
#   topic   - restates the note's subject in other words.
#   buried  - a fact that appears once in the body and is ABSENT from the
#             note's name and description.
#   oblique - a symptom or situation, with the mechanism not named.
QUERY_KINDS = ("topic", "buried", "oblique")


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    kind: str
    expect_notes: tuple[str, ...]
    why: str

    @property
    def relevant(self) -> set[str]:
        return set(self.expect_notes)


def load_query_set(path: Path, notes: list[Note]) -> list[Query]:
    """Parse `path` and check every expectation against the loaded corpus.

    An expected note id that is not in the corpus raises rather than scoring
    zero. A typo'd or renamed path would otherwise look exactly like a
    retrieval failure and silently drag both strategies' recall down -- the
    single most plausible way for this harness to report a confident wrong
    answer.
    """
    if not path.is_file():
        raise QuerySetError(f"query set not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise QuerySetError(f"{path}: unparseable YAML: {exc}") from exc
    if not isinstance(document, dict) or "queries" not in document:
        raise QuerySetError(
            f"{path}: expected a mapping with a top-level 'queries:' list"
        )
    raw_queries = document["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise QuerySetError(f"{path}: 'queries' must be a non-empty list")

    known = {note.note_id for note in notes}
    queries: list[Query] = []
    seen_ids: set[str] = set()
    problems: list[str] = []

    for position, entry in enumerate(raw_queries, start=1):
        where = f"{path}: queries[{position}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: expected a mapping")
            continue
        query_id = str(entry.get("id") or f"q{position:02d}")
        text = str(entry.get("query") or "").strip()
        if not text:
            problems.append(f"{where} ({query_id}): 'query' is empty or missing")
        if query_id in seen_ids:
            problems.append(f"{where}: duplicate id {query_id!r}")
        seen_ids.add(query_id)

        kind = str(entry.get("kind") or "").strip().lower()
        if kind not in QUERY_KINDS:
            problems.append(
                f"{where} ({query_id}): 'kind' must be one of "
                f"{', '.join(QUERY_KINDS)}, got {kind or '<missing>'!r}"
            )

        expected = entry.get("expect_notes") or entry.get("expect") or []
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(expected, list) or not expected:
            problems.append(
                f"{where} ({query_id}): 'expect_notes' must list at least one note id"
            )
            expected = []
        expected = [str(item).strip() for item in expected]
        for note_id in expected:
            if note_id not in known:
                problems.append(
                    f"{where} ({query_id}): expects {note_id!r}, which is not in "
                    f"the corpus"
                )

        queries.append(
            Query(
                query_id=query_id,
                text=text,
                kind=kind,
                expect_notes=tuple(expected),
                why=str(entry.get("why") or "").strip(),
            )
        )

    if problems:
        raise QuerySetError(
            "query set is invalid:\n  " + "\n  ".join(problems)
        )
    return queries


def kind_mix(queries: list[Query]) -> dict[str, int]:
    """How many cases of each kind, in QUERY_KINDS order.

    Reported with every run so the mix behind an aggregate is on the page
    next to it rather than something a reader has to go and count.
    """
    return {
        kind: sum(1 for query in queries if query.kind == kind)
        for kind in QUERY_KINDS
    }
