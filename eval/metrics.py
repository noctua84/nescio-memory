"""recall@k and MRR@10 over a ranked list of retrieved units.

THE ONE JUDGEMENT CALL IN HERE, stated plainly because it changes the answer:

`k` counts retrieved UNITS, not distinct notes. A top-5 for the `chunk`
strategy means the five best-scoring chunks, which may all belong to two notes;
a top-5 for `summary` is always five distinct notes. recall@k is then "was a
note I wanted among the parents of those k units".

That is production's semantics -- /search takes top_k and returns that many
`learnings` rows, each a chunk -- and it is also the consumer's real question:
given k result slots, do I get the note I need? It does mean chunking is
charged for spending result slots on several chunks of the same note. That
penalty is a real property of the strategy, not an artefact, so it is measured
rather than normalized away; `distinct_notes_at_k` is reported alongside so a
reader can see exactly how much slot dilution is driving any gap.

The alternative framing -- dedupe to notes first, then take the top k notes --
would flatter chunking by giving it an unlimited candidate budget that
production does not give it. If you ever want that number, it is a different
metric and belongs beside this one, not instead of it.
"""
from dataclasses import dataclass

DEFAULT_KS = (1, 3, 5, 10)
MRR_CUTOFF = 10


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    first_relevant_rank: int | None
    distinct_notes_at_k: dict[int, int]
    retrieved_note_ids: list[str]


def score_query(
    query_id: str,
    hit_note_ids: list[str],
    relevant_note_ids: set[str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> QueryScore:
    """Score one query.

    `hit_note_ids` is the parent note of each retrieved unit, in rank order and
    WITH duplicates -- the duplicates are the point (see the module docstring).
    """
    if not relevant_note_ids:
        raise ValueError(f"query {query_id!r} lists no expected notes")

    recall: dict[int, float] = {}
    distinct: dict[int, int] = {}
    for k in ks:
        parents = set(hit_note_ids[:k])
        distinct[k] = len(parents)
        recall[k] = len(parents & relevant_note_ids) / len(relevant_note_ids)

    first_rank: int | None = None
    for position, note_id in enumerate(hit_note_ids[:MRR_CUTOFF], start=1):
        if note_id in relevant_note_ids:
            first_rank = position
            break

    return QueryScore(
        query_id=query_id,
        recall_at_k=recall,
        reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
        first_relevant_rank=first_rank,
        distinct_notes_at_k=distinct,
        retrieved_note_ids=hit_note_ids,
    )


@dataclass(frozen=True)
class StrategyScore:
    strategy_name: str
    queries: int
    recall_at_k: dict[int, float]
    mrr_at_10: float
    mean_distinct_notes_at_k: dict[int, float]
    per_query: list[QueryScore]


def aggregate(
    strategy_name: str,
    scores: list[QueryScore],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> StrategyScore:
    """Macro-average over queries: every query weighs the same.

    Macro rather than micro (pooling all relevant notes) so a single query that
    lists six expected notes cannot dominate ten queries that list one each.
    """
    if not scores:
        raise ValueError("no query scores to aggregate")
    count = len(scores)
    return StrategyScore(
        strategy_name=strategy_name,
        queries=count,
        recall_at_k={
            k: sum(score.recall_at_k[k] for score in scores) / count for k in ks
        },
        mrr_at_10=sum(score.reciprocal_rank for score in scores) / count,
        mean_distinct_notes_at_k={
            k: sum(score.distinct_notes_at_k[k] for score in scores) / count
            for k in ks
        },
        per_query=scores,
    )
