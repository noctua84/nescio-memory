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
    # "" for the overall score; one of eval.queries.QUERY_KINDS for a
    # per-class slice. Carried on the score rather than only in the dict key
    # so a score passed around on its own still says what it covers.
    kind: str
    recall_at_k: dict[int, float]
    mrr_at_10: float
    mean_distinct_notes_at_k: dict[int, float]
    per_query: list[QueryScore]


def aggregate(
    strategy_name: str,
    scores: list[QueryScore],
    ks: tuple[int, ...] = DEFAULT_KS,
    kind: str = "",
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
        kind=kind,
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


def aggregate_by_kind(
    strategy_name: str,
    scores: list[QueryScore],
    kinds: dict[str, str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, StrategyScore]:
    """The same aggregate, sliced by the query set's `kind` field.

    WHY THIS IS NOT OPTIONAL. The overall figure is a weighted average of these
    slices with the case mix as the weights, so on its own it is not a finding
    about retrieval at all -- it is a finding about the query set. Two honest
    sets over the same corpus, differing only in how many `buried` cases they
    contain, will report different winners. The per-class numbers are what
    survive a change of mix, and they are what makes a claim like "summary
    loses when the query names a symptom rather than the mechanism" something a
    reader can check rather than take.

    An absent kind is simply absent from the result; it is not reported as a
    class with zero queries, which would read as a measured failure rather than
    an unmeasured case. A query_id missing from `kinds` raises, because
    silently dropping a case would shift every slice.
    """
    grouped: dict[str, list[QueryScore]] = {}
    for score in scores:
        try:
            kind = kinds[score.query_id]
        except KeyError as exc:
            raise ValueError(
                f"query {score.query_id!r} has no kind; every case must declare "
                f"one or the per-class slices are built on a partial set"
            ) from exc
        grouped.setdefault(kind, []).append(score)
    return {
        kind: aggregate(strategy_name, group, ks, kind=kind)
        for kind, group in grouped.items()
    }
