"""The hybrid search statement can actually use its indexes.

tests/conftest.py forces exact scans suite-wide (`enable_indexscan = off`), so
no other test proves that the statement production runs -- the one
app/repositories/passage_search.py:hybrid_statement builds -- is still shaped
so that its vector arm can be served by the HNSW index `passages_embedding_idx`
and its full-text arm by the GIN index `ix_passages_tsv`. A refactor that, say,
adds a tiebreak to the vector arm's ORDER BY, wraps the distance in an
expression, or moves a predicate onto `notes` would leave every result-level
test green while quietly turning both arms into full-table scans.

What this guards, and what it does not. The planner is FORCED here
(`enable_seqscan` and `enable_sort` off, index scans back on) rather than left
to win on cost: measured on the dev machine, HNSW only wins unforced at around
20,000 1024-d rows, and seeding that took ~60s. Forcing makes the test fast and
deterministic, and it still fails on exactly the regression it exists for --
if an arm's query shape makes its index unusable, no amount of forcing can put
that index in the plan, and the planner falls back to a disabled path instead.
It does NOT prove the index wins on cost at production scale.

`enable_bitmapscan` stays ON deliberately: GIN supports only bitmap scans, so
turning bitmap scans off would make ix_passages_tsv unusable by construction.
The vector arm still cannot drift to a bitmap scan on ix_passages_client_note,
because that path needs a Sort to produce distance order and sorting is
disabled, while the HNSW scan yields rows in `<=>` order with no Sort at all.

Every planner setting is SET LOCAL inside the `connection` fixture's outer
transaction, which is always rolled back, so nothing leaks into other tests.
"""
import json
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from app.config import settings
from app.repositories.passage_search import hybrid_statement
from tests.factories import make_note
from tests.fakes import fake_embedding

HNSW_INDEX = "passages_embedding_idx"
GIN_INDEX = "ix_passages_tsv"

# A dedicated tenant that owns the whole corpus, so `client_name` is not
# selective and ix_passages_client_note cannot narrow either arm cheaply.
TENANT = "explain-tenant"
SCOPE = "explain_repo"
CORPUS_ROWS = 2_000
# A lexeme no other passage contains, on 1 row in RARE_EVERY, so the GIN index
# is the cheapest way to find the FTS arm's matches.
RARE_LEXEME = "quokkaflux"
RARE_EVERY = 100
REPETITIONS = 3


def explain_json(db: Session, statement: Any) -> dict[str, Any]:
    """EXPLAIN (FORMAT JSON) `statement` exactly as production would execute
    it, and return the root plan node.

    The prefix is added in a before_cursor_execute listener rather than by
    compiling the statement to a string: by then SQLAlchemy has run the Vector
    type's bind processor on the query vector, so the parameters EXPLAIN sees
    are exactly the ones a real search sends.
    """
    connection = db.connection()

    def prefix_explain(conn, cursor, statement_sql, parameters, context, executemany):
        return "EXPLAIN (FORMAT JSON) " + statement_sql, parameters

    event.listen(connection, "before_cursor_execute", prefix_explain, retval=True)
    try:
        result = connection.execute(statement)
        plan = result.cursor.fetchone()[0]
        result.close()
    finally:
        event.remove(connection, "before_cursor_execute", prefix_explain)
    if isinstance(plan, str):
        plan = json.loads(plan)
    return plan[0]["Plan"]


def walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every node of the plan subtree rooted at `node`, root first."""
    yield node
    for child in node.get("Plans", []):
        yield from walk(child)


def _own_text(node: dict[str, Any]) -> str:
    """The node's own fields (conditions, sort keys, order-by), not its children's."""
    return json.dumps({k: v for k, v in node.items() if k != "Plans"})


def arm_subtree(root: dict[str, Any], marker: str) -> list[dict[str, Any]]:
    """The nodes of the arm identified by `marker` (`<=>` or `@@`).

    The arm's root is the topmost node, in breadth-first order, whose OWN
    fields mention the marker: an HNSW Index Scan's Order By, a Sort Key over
    the distance, a Bitmap Heap Scan's Recheck Cond, a Seq Scan's Filter. Its
    subtree is the arm. Identified by expression, never by plan position, so
    a reshuffled join order cannot point the assertion at the wrong arm.
    """
    queue = [root]
    while queue:
        node = queue.pop(0)
        if marker in _own_text(node):
            return list(walk(node))
        queue.extend(node.get("Plans", []))
    return []


def summarise(root: dict[str, Any], depth: int = 0) -> str:
    """A compact plan outline for assertion messages: node types, indexes and
    relations only. The full JSON carries a 1024-float vector literal."""
    label = root["Node Type"]
    for key in ("Index Name", "Relation Name", "Alias"):
        if key in root:
            label += f" [{key}={root[key]}]"
    lines = ["  " * depth + label]
    for child in root.get("Plans", []):
        lines.append(summarise(child, depth + 1))
    return "\n".join(lines)


def uses_index(nodes: list[dict[str, Any]], node_types: set[str], index: str) -> bool:
    return any(
        node["Node Type"] in node_types and node.get("Index Name") == index
        for node in nodes
    )


def _seed_corpus(db: Session) -> None:
    """Seed CORPUS_ROWS passages for TENANT, building both indexes once.

    Dropping the indexes and recreating them after the insert is the same
    bulk-build trick tests/test_search_recall.py uses for HNSW: one build
    instead of thousands of incremental insertions. Each definition is read
    from pg_indexes so it cannot drift from the migration, and DDL is
    transactional, so the rollback restores the real indexes.

    The GIN index is rebuilt for a second reason, measured: inserted
    incrementally, every new entry sits in GIN's pending list (fastupdate is
    on by default), which a search must scan in full and the planner costs
    accordingly. ANALYZE does not flush it. With the pending list holding the
    whole corpus, the planner preferred a bitmap scan on
    ix_passages_client_note with `@@` as a Filter, at 2,000 rows unforced too.
    A freshly built GIN index has no pending list and is what a vacuumed
    production table looks like.
    """
    indexdefs = {}
    for name in (HNSW_INDEX, GIN_INDEX):
        indexdefs[name] = db.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
            {"name": name},
        ).scalar()
        assert indexdefs[name], f"{name} not found in pg_indexes -- nothing to rebuild"
        db.execute(text(f"DROP INDEX {name}"))

    note = make_note(db, TENANT, scope=SCOPE, path_or_uri="explain/corpus.md")
    rows = []
    for ordinal in range(CORPUS_ROWS):
        content = f"explain passage {ordinal}"
        if ordinal % RARE_EVERY == 0:
            content += f" {RARE_LEXEME}"
        rows.append(
            {
                "note_id": note.id,
                "client_name": TENANT,
                "scope": SCOPE,
                "ordinal": ordinal,
                "content": content,
                "embedding": str(fake_embedding(content)),
            }
        )
    # One executemany instead of make_passage's flush per row. tsv is built
    # with to_tsvector in SQL: only the B section matters to this corpus, and
    # what is under test is whether the GIN index is used, not ranking.
    db.execute(
        text(
            "INSERT INTO passages "
            "(note_id, client_name, scope, ordinal, content, embedding, tsv) "
            "VALUES (:note_id, :client_name, :scope, :ordinal, :content, "
            "CAST(:embedding AS vector), "
            "setweight(to_tsvector('english', :content), 'B'))"
        ),
        rows,
    )
    for indexdef in indexdefs.values():
        db.execute(text(indexdef))
    db.execute(text("ANALYZE passages"))

    # Planner forcing, transaction-local: see the module docstring.
    db.execute(text("SET LOCAL enable_indexscan = on"))
    db.execute(text("SET LOCAL enable_seqscan = off"))
    db.execute(text("SET LOCAL enable_sort = off"))


@pytest.mark.parametrize("repo_filter", [None, SCOPE], ids=["unscoped", "scoped"])
@pytest.mark.parametrize("repetition", range(REPETITIONS))
def test_each_arm_of_the_hybrid_statement_uses_its_index(
    db_session, repetition, repo_filter
):
    """The vector arm is an Index Scan on the HNSW index; the FTS arm reads
    the GIN index. Parametrised over REPETITIONS fresh seeds and index builds,
    so a plan that only holds some of the time fails here."""
    _seed_corpus(db_session)

    statement = hybrid_statement(
        TENANT,
        RARE_LEXEME,
        fake_embedding("a query about nothing in particular"),
        10,
        repo_filter,
        candidates=settings.search_candidates,
        rrf_k=settings.rrf_k,
        fts_weight=1.0,
        vector_weight=1.0,
    )
    root = explain_json(db_session, statement)
    outline = summarise(root)

    vector_arm = arm_subtree(root, "<=>")
    assert vector_arm, f"no plan node mentions `<=>` -- vector arm missing:\n{outline}"
    assert uses_index(vector_arm, {"Index Scan"}, HNSW_INDEX), (
        f"the vector arm does not use an Index Scan on {HNSW_INDEX}, so the "
        "statement's shape no longer lets the HNSW index serve it. Plan:\n"
        f"{outline}"
    )

    fts_arm = arm_subtree(root, "@@")
    assert fts_arm, f"no plan node mentions `@@` -- FTS arm missing:\n{outline}"
    assert uses_index(
        fts_arm, {"Bitmap Index Scan", "Index Scan"}, GIN_INDEX
    ), (
        f"the FTS arm does not read {GIN_INDEX}, so the statement's shape no "
        f"longer lets the GIN index serve it. Plan:\n{outline}"
    )

    # The arms are distinct subtrees: neither index turns up in the wrong arm.
    assert not uses_index(fts_arm, {"Index Scan"}, HNSW_INDEX), outline
    assert not uses_index(
        vector_arm, {"Bitmap Index Scan", "Index Scan"}, GIN_INDEX
    ), outline
