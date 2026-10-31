"""Passage hybrid search. Reads only: ingest and passage writes live in note.py.

Tenant discipline, as stated in app/repositories/note.py's module docstring and
held to here: every statement carries `client_name == client_name` as a
DIRECT argument to `.where(...)` -- top level, never only inside a join
condition, never inside an `or_()` group. A filter inside a composed predicate
is correct only while every branch of the composition carries it, and one
refactor can drop it from a single branch while the rest still look right.
At the top level each table gets one predicate that cannot be partially
applied, and no id a caller obtained elsewhere is authorisation to read the
row it names.

hybrid_statement touches `passages` once per arm and once more, with `notes`,
in the final join. Every one of those SELECTs carries its own top-level
client_name predicate: each arm's candidate set is tenant-scoped before it is
ranked, and the final join filters both tables again, so either one alone
still enforces the boundary.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import (
    Double,
    Row,
    Select,
    Text,
    and_,
    bindparam,
    cast,
    func,
    literal,
    literal_column,
    select,
    union_all,
)
from sqlalchemy.orm import Session

from app.config import settings
from app.models.note import Note
from app.models.passage import Passage
from app.repositories.search_guards import _apply_search_guards

_FTS_ARM = "fts"
_VECTOR_ARM = "vec"


def hybrid_statement(
    client_name: str,
    query: str,
    query_vector: Sequence[float] | None,
    top_k: int,
    repo_filter: str | None,
    *,
    candidates: int,
    rrf_k: int,
    fts_weight: float,
    vector_weight: float,
) -> Select[Any]:
    """Build the single hybrid search statement: a full-text arm and a vector
    arm, fused by weighted reciprocal rank fusion, joined to note metadata.

    Pure: it builds a Select and executes nothing, so a test can compile or
    EXPLAIN exactly what production runs. An arm is included only when its
    weight is > 0 (and, for the vector arm, a query vector exists); with
    neither included there is nothing to rank and ValueError is raised.

    Each arm contributes at most n = max(candidates, top_k) rows, so a caller
    asking for more than `candidates` results is never capped below top_k.
    """
    n = max(candidates, top_k)
    arms = []

    if fts_weight > 0:
        q = func.websearch_to_tsquery("english", query)
        # Left on the default weight vector {0.1, 0.2, 0.4, 1.0} deliberately:
        # putting description at weight C depends on those defaults, so a
        # custom vector here would silently rebalance A/B/C.
        r_fts = func.ts_rank_cd(Passage.tsv, q).label("r_fts")
        fts_inner = (
            select(Passage.id, r_fts)
            .where(
                Passage.client_name == client_name,
                Passage.tsv.op("@@")(q),
            )
            # Passage.id is a deterministic tiebreak: ts_rank_cd ties are
            # common and an unstable order makes tests flap.
            .order_by(r_fts.desc(), Passage.id)
            .limit(n)
        )
        if repo_filter:
            # passages.scope, not notes.scope: a single-table predicate on the
            # table the GIN index scans (see app/models/passage.py).
            fts_inner = fts_inner.where(Passage.scope == repo_filter)
        fts = fts_inner.subquery("fts_candidates")
        arms.append(
            select(
                fts.c.id,
                func.row_number()
                .over(order_by=(fts.c.r_fts.desc(), fts.c.id))
                .label("r"),
                literal(float(fts_weight), Double).label("w"),
                literal(_FTS_ARM, Text).label("arm"),
            )
        )

    if query_vector is not None and vector_weight > 0:
        # Bound through the column's own Vector type, never rendered as a
        # literal: the engine runs with hide_parameters, and a 1024-float
        # literal would also put the query's embedding into every logged
        # statement. One named bindparam, reused, so the vector is sent once.
        qv = bindparam("query_vector", value=list(query_vector),
                       type_=Passage.embedding.type)
        d = Passage.embedding.cosine_distance(qv).label("d")  # the <=> operator

        # #42/#53: a stored embedding can be zero-norm even though the vector
        # passed validation before it was written -- pgvector's float4 storage
        # can flatten components Python's float64 check accepted (all-subnormal,
        # or squares that underflow float32 before summing). Against a real
        # query only the all-zero shape comes back as NaN distance; the others
        # are clamped to 0.0 and would rank as a PERFECT match. So the reliable
        # exclusion is embedding_degenerate, a generated column holding the
        # stored row's self-distance-is-NaN test, computed once by Postgres at
        # write time instead of per candidate per query (measured at 29-35% of
        # the removed learnings API's search time when it was computed per
        # query). `d != NaN` is kept as
        # cheap defence -- the query vector is validated before it gets here,
        # so it should never itself exclude a row -- and since Postgres treats
        # NaN = NaN as TRUE it removes exactly the NaN rows. Both are filters
        # applied after the index produces candidates, like client_name; they
        # do not change ORDER BY or LIMIT.
        vec_inner = (
            select(Passage.id, d)
            .where(
                Passage.client_name == client_name,
                Passage.embedding_degenerate.is_(False),
                d != literal_column("'NaN'::float8"),
            )
            # Distance ONLY. The HNSW index produces rows in
            # `embedding <=> :qv` order and nothing else, so an ORDER BY of
            # exactly that is satisfied by the index scan feeding LIMIT
            # directly. A Passage.id tiebreak is not: it puts a sort above the
            # index scan (an Incremental Sort on pg17, measured) and raises the
            # arm's estimated cost, which moves the planner toward an exact
            # scan-and-sort. Ties are broken deterministically in the outer
            # row_number below, over just these n rows, where it is free.
            .order_by(d)
            .limit(n)
        )
        if repo_filter:
            vec_inner = vec_inner.where(Passage.scope == repo_filter)
        vec = vec_inner.subquery("vec_candidates")
        arms.append(
            select(
                vec.c.id,
                func.row_number().over(order_by=(vec.c.d, vec.c.id)).label("r"),
                literal(float(vector_weight), Double).label("w"),
                literal(_VECTOR_ARM, Text).label("arm"),
            )
        )

    if not arms:
        # Unreachable through settings (a validator forbids both weights being
        # 0, and the caller substitutes an FTS weight when it has no vector),
        # but a statement with no arm would be malformed SQL, not an empty
        # result, so refuse it by name.
        raise ValueError(
            "hybrid_statement needs at least one arm: fts_weight must be > 0, "
            "or vector_weight must be > 0 with a query_vector"
        )

    ranked = (arms[0] if len(arms) == 1 else union_all(*arms)).subquery("ranked")

    # Weighted RRF: score = SUM over arms of w / (rrf_k + rank). Both operands
    # are cast to double precision explicitly: rrf_k + r is bigint, and if w
    # ever reached the division as an integer the result would be integer
    # division and every score would silently be 0.
    score = func.sum(
        cast(ranked.c.w, Double) / cast(rrf_k + ranked.c.r, Double)
    ).label("score")
    # A passage found by only one arm has a NULL rank for the other, which is
    # also what a weight-0 (excluded) arm reports for every row.
    fts_rank = func.min(ranked.c.r).filter(ranked.c.arm == _FTS_ARM).label("fts_rank")
    vector_rank = (
        func.min(ranked.c.r).filter(ranked.c.arm == _VECTOR_ARM).label("vector_rank")
    )
    fused = (
        select(ranked.c.id, score, fts_rank, vector_rank)
        .group_by(ranked.c.id)
        .subquery("fused")
    )

    return (
        select(
            Passage.id, Passage.note_id, Passage.ordinal,
            Passage.heading_path, Passage.content,
            fused.c.score, fused.c.fts_rank, fused.c.vector_rank,
            Note.name, Note.description, Note.type, Note.path_or_uri,
            Note.scope, Note.provenance_kind, Note.provenance_date,
            Note.status,
        )
        .select_from(fused)
        .join(Passage, Passage.id == fused.c.id)
        .join(
            Note,
            and_(
                Note.id == Passage.note_id,
                Note.client_name == Passage.client_name,
            ),
        )
        # Both client_name predicates are present on purpose: the one on
        # Passage re-checks every fused id against the tenant (an id is never
        # authorisation on its own), and the one on Note is independent of the
        # join condition, so dropping either still leaves the boundary enforced.
        .where(
            Passage.client_name == client_name,
            Note.client_name == client_name,
        )
        # Passage.id is a deterministic tiebreak: equal fused scores are
        # routine (e.g. two passages each found by one arm at the same rank).
        .order_by(fused.c.score.desc(), Passage.id)
        .limit(top_k)
    )


class PassageSearchRepository:
    """Passage search over the notes/passages tables. Reads only and commits
    nothing: the route owns the transaction boundary."""

    def __init__(self, db: Session, client_name: str):
        self.db = db
        self.client_name = client_name

    def search(
        self,
        query: str,
        query_vector: Sequence[float] | None,
        top_k: int,
        repo_filter: str | None = None,
    ) -> Sequence[Row[Any]]:
        """Hybrid search over passages, carrying their note's metadata.

        `query_vector` is None when the query could not be embedded (degraded
        mode) or when VECTOR_WEIGHT=0 means it was never requested; either way
        only the FTS arm runs.

        `settings` is read at CALL time, not captured at import, so tests can
        monkeypatch it.
        """
        fts_weight = settings.fts_weight
        if query_vector is None:
            # Decision P3: with no vector the FTS arm runs alone, and at
            # FTS_WEIGHT=0 every fused score would be 0 -- an arbitrary order
            # dressed up as a ranking. Any positive weight ranks identically
            # when it is the only arm, so 1.0 stands in.
            fts_weight = fts_weight or 1.0

        # Unconditionally, FTS-only included: statement_timeout caps every
        # search, and the read-back guard is what proves it was applied.
        _apply_search_guards(self.db)

        stmt = hybrid_statement(
            self.client_name, query, query_vector, top_k, repo_filter,
            candidates=settings.search_candidates,
            rrf_k=settings.rrf_k,
            fts_weight=fts_weight,
            vector_weight=settings.vector_weight,
        )
        return self.db.execute(stmt).all()
