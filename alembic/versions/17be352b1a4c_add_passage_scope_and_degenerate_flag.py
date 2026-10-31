"""add passage scope and degenerate flag

Revision ID: 17be352b1a4c
Revises: 2eb0691f043c
Create Date: 2026-10-31 00:00:00.000000

Adds two columns to `passages` that hybrid passage search needs:

* `scope`, copied from the owning note, so the scope predicate is a
  single-table predicate on the table the HNSW index scans.
* `embedding_degenerate`, a STORED generated flag for zero-norm vectors, the
  same guard `learnings` got in 9439c6362bb3.
"""
from typing import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '17be352b1a4c'
down_revision: str | Sequence[str] | None = '2eb0691f043c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # scope is duplicated onto passages for the same reason client_name is
    # (2eb0691f043c): the filter must be a single-table predicate on the table
    # actually being scanned. notes.scope would have to be applied after
    # joining the HNSW candidates back to notes, and a post-join filter on
    # HNSW candidates starves the result set -- the class of issues #12/#74.
    # The composite FK below makes Postgres, not application code, guarantee
    # that the copy never drifts from the note's own scope.
    #
    # Nullable first so the backfill can run; NOT NULL is set in the same
    # ALTER as the generated column (see below). The join on client_name as
    # well as id mirrors the composite FK it is about to be replaced by.
    op.execute("ALTER TABLE passages ADD COLUMN scope text")
    op.execute(
        """
        UPDATE passages p
        SET scope = n.scope
        FROM notes n
        WHERE n.id = p.note_id AND n.client_name = p.client_name
        """
    )

    # embedding_degenerate is computed by Postgres from the STORED vector,
    # never by application code. pgvector's float4 storage can turn a vector
    # that Python's own validation accepted into a zero-norm vector once
    # written (#42/#53), so the flag must be derived from what is actually on
    # disk. Without it, a zero-norm stored vector makes `<=>` return NaN,
    # which would score as a perfect match. `<=>` (cosine_distance) is
    # IMMUTABLE, which is required for a STORED generated column.
    #
    # One ALTER TABLE for SET NOT NULL and ADD COLUMN ... STORED, rather than
    # separate statements. ADD COLUMN ... STORED rewrites the whole table
    # under an ACCESS EXCLUSIVE lock -- every row is read and rewritten to
    # compute the new column -- and, because this table also carries the HNSW
    # index `passages_embedding_idx`, that rewrite rebuilds every index on the
    # table too. The index rebuild is the dominant cost, not the column
    # computation. Splitting the statements would rewrite the heap and rebuild
    # the index more than once for no benefit. That is acceptable at
    # self-hosted sizes; on a very large table, run this during a maintenance
    # window.
    #
    # WIDTH CHANGE WARNING: a future migration that changes embedding's
    # vector width (as 57cfbdfa0d6a and 58f3ecef1edf did for learnings)
    # cannot run `ALTER COLUMN embedding TYPE vector(N)` while
    # embedding_degenerate exists -- Postgres refuses to alter the type of a
    # column a generated column reads. Such a migration must DROP
    # embedding_degenerate first, alter embedding's type, then re-add
    # embedding_degenerate with this same generation expression.
    op.execute(
        """
        ALTER TABLE passages
            ALTER COLUMN scope SET NOT NULL,
            ADD COLUMN embedding_degenerate boolean
                GENERATED ALWAYS AS (((embedding <=> embedding) = 'NaN'::float8)) STORED
                NOT NULL
        """
    )

    # A composite FK's referenced columns must be covered by a unique
    # constraint.
    op.create_unique_constraint(
        "uq_notes_id_client_scope", "notes", ["id", "client_name", "scope"]
    )
    op.drop_constraint("fk_passages_note", "passages", type_="foreignkey")
    op.create_foreign_key(
        "fk_passages_note",
        "passages",
        "notes",
        ["note_id", "client_name", "scope"],
        ["id", "client_name", "scope"],
        ondelete="CASCADE",
    )
    # Superseded: uq_notes_id_client_scope covers (id, client_name) too, and
    # nothing references the narrower constraint any more.
    op.drop_constraint("uq_notes_id_client", "notes", type_="unique")


def downgrade() -> None:
    # Opposite order of upgrade(): the narrower unique constraint must exist
    # before the 2-column FK can reference it, and the 3-column FK must go
    # before uq_notes_id_client_scope can be dropped.
    op.create_unique_constraint("uq_notes_id_client", "notes", ["id", "client_name"])
    op.drop_constraint("fk_passages_note", "passages", type_="foreignkey")
    op.create_foreign_key(
        "fk_passages_note",
        "passages",
        "notes",
        ["note_id", "client_name"],
        ["id", "client_name"],
        ondelete="CASCADE",
    )
    op.drop_constraint("uq_notes_id_client_scope", "notes", type_="unique")

    # A single ALTER TABLE: DROP COLUMN does not rewrite the table (it only
    # marks the column dropped in the catalog), so combining them is for
    # symmetry with upgrade() and to take the lock once.
    op.execute(
        """
        ALTER TABLE passages
            DROP COLUMN embedding_degenerate,
            DROP COLUMN scope
        """
    )
