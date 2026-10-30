"""All notes/passages SQL (v2). Nothing outside this module touches those tables.

Tenant discipline, stated once so it is not re-argued per method: every statement
here carries `client_name == self.client_name` as a DIRECT argument to
`.where(...)` -- top level, never only inside a join condition, never inside an
`or_()` group. Same reasoning as app/repositories/learning.py:305-319: a filter
inside a composed predicate is correct only while every branch of that
composition carries it, so one `continue` or one refactored predicate list leaks
another tenant's rows from a single branch while the rest still look right --
invisible in review and in any test that misses that branch. At the top level
there is one filter, it cannot be partially applied, and no id a caller obtained
elsewhere is authorisation to read the row it names.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Row, and_, bindparam, delete, func, insert, literal_column, select, text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import settings
from app.models.note import Note
from app.models.passage import Passage

# Columns upsert_note overwrites on conflict. The identity columns it conflicts
# ON, and created_at, are deliberately absent.
_MUTABLE_NOTE_COLUMNS = (
    "name", "description", "type", "provenance_kind", "provenance_date",
    "status", "content_hash", "pipeline_version", "body", "meta",
)


@dataclass(frozen=True)
class LockedNote:
    """A pre-existing note's skip key, as read under the row lock."""

    id: int
    content_hash: str
    pipeline_version: str


@dataclass(frozen=True)
class PassageRow:
    """One passage ready to be written: text, vector and both FTS leads.

    `fts_a`/`fts_c` come from app.core.pipeline.fts_lead_a/fts_lead_c -- already
    composed and NULL-free; see replace_passages for why that matters.
    """

    ordinal: int
    heading_path: str | None
    content: str
    embedding: list[float]
    fts_a: str
    fts_c: str


def _weighted(param: str, label: str) -> Any:
    """`setweight(to_tsvector('english', :param), 'label')`.

    The label goes through literal_column, not a bound parameter: setweight's
    second argument is Postgres's `"char"`, the driver renders a bound string
    as VARCHAR, and there is no such overload ("function setweight(tsvector,
    character varying) does not exist"). An unadorned SQL literal is resolved
    to `"char"` by Postgres itself; only three fixed labels are passed in.
    """
    return func.setweight(
        func.to_tsvector("english", bindparam(param)), literal_column(f"'{label}'")
    )


class NoteRepository:
    """All notes/passages access lives here. Commits nothing: the route owns
    the transaction boundary, so a note and its passages land together."""

    def __init__(self, db: Session, client_name: str):
        self.db = db
        self.client_name = client_name

    def lock_note(self, scope: str, path_or_uri: str) -> LockedNote | None:
        """Lock this note's row, if it exists, and read its skip key.

        `FOR UPDATE` is what serialises concurrent ingests of the same note:
        the second blocks here until the first commits, then sees the hash the
        first wrote instead of racing it. None means the note does not exist
        yet -- there is no row to lock, and the caller must treat it as new.
        """
        stmt = (
            select(Note.id, Note.content_hash, Note.pipeline_version)
            .where(
                Note.client_name == self.client_name,
                Note.scope == scope,
                Note.path_or_uri == path_or_uri,
            )
            .with_for_update()
        )
        row = self.db.execute(stmt).one_or_none()
        if row is None:
            return None
        return LockedNote(row.id, row.content_hash, row.pipeline_version)

    def upsert_note(
        self,
        scope: str,
        path_or_uri: str,
        *,
        name: str | None,
        description: str | None,
        type_: str | None,
        provenance_kind: str | None,
        provenance_date: date | None,
        status: str | None,
        content_hash: str,
        pipeline_version: str,
        body: str,
        meta: dict[str, Any],
    ) -> int:
        """Insert or update the note by its (client, scope, path) identity.

        `client_name` is stamped from self, never passed by the caller --
        exactly as LearningRepository.add stamps it, so no route can choose
        which tenant it writes to.

        Why this is a SECOND statement after lock_note, which looks redundant
        and is not: one INSERT ... ON CONFLICT DO UPDATE ... RETURNING cannot
        tell the caller whether the row pre-existed, nor what its OLD
        content_hash was -- `excluded` is not available in RETURNING and
        `notes.content_hash` there is the post-update value -- and without the
        old hash there is no versioned skip. The `xmax <> 0` idiom was
        rejected: it is obscure, and xmax is also non-zero for a row merely
        locked by another transaction, so it reports "inserted" for rows that
        were not. With two statements the one remaining race -- a concurrent
        inserter between the SELECT and the INSERT -- makes lock_note return
        None, the caller treat the note as new, and the passages be rewritten
        redundantly. Never wrongly; the ON CONFLICT clause is what makes that
        race a redundant rewrite rather than an IntegrityError.
        """
        stmt = pg_insert(Note).values(
            client_name=self.client_name,
            scope=scope,
            path_or_uri=path_or_uri,
            name=name,
            description=description,
            type=type_,
            provenance_kind=provenance_kind,
            provenance_date=provenance_date,
            status=status,
            content_hash=content_hash,
            pipeline_version=pipeline_version,
            body=body,
            meta=meta,
        )
        set_: dict[str, Any] = {
            column: stmt.excluded[column] for column in _MUTABLE_NOTE_COLUMNS
        }
        # Explicit, because the ORM-level `onupdate=func.now()` on the model
        # does not fire for a Core ON CONFLICT DO UPDATE.
        set_["updated_at"] = func.now()
        stmt = stmt.on_conflict_do_update(
            constraint="uq_notes_client_scope_path", set_=set_
        ).returning(Note.id)
        return self.db.execute(stmt).scalar_one()

    def replace_passages(self, note_id: int, passages: Sequence[PassageRow]) -> int:
        """Delete this note's passages and write the given set in their place.

        Wholesale replacement rather than a diff: ordinals shift whenever a
        body changes, so there is no stable per-passage identity to update
        against. Returns the number inserted. Does not commit.
        """
        # The tenant filter is here even though the composite FK
        # (note_id, client_name) -> notes(id, client_name) already guarantees
        # every passage under this note belongs to this tenant: a filter
        # correct only because another constraint is correct is one migration
        # away from being wrong. It is free too -- client_name leads
        # ix_passages_client_note.
        self.db.execute(
            delete(Passage).where(
                Passage.client_name == self.client_name,
                Passage.note_id == note_id,
            )
        )

        if not passages:
            return 0

        # Three weighted sections: A = note name + heading path, B = passage
        # content, C = note description. These weights, this expression and
        # app/core/pipeline.py's fts_lead_a/fts_lead_c are ONE contract,
        # versioned by that module's FTS_VERSION -- tsv is materialised here at
        # ingest, so a change to any of them is invisible to already-stored
        # rows and needs a re-ingest. Bump FTS_VERSION if you touch this.
        #
        # The A and C strings arrive already composed and NULL-free from
        # Python. Doing it in SQL as `name || ' ' || heading_path` would yield
        # NULL for any note without a name -- 27 of the 574 corpus notes have
        # no frontmatter at all -- silently blanking that whole weight section.
        #
        # D is left unused: ts_rank_cd's default weight vector
        # {D, C, B, A} = {0.1, 0.2, 0.4, 1.0} is what search_fts relies on.
        tsv = (
            _weighted("fts_a", "A")
            .op("||")(_weighted("content", "B"))
            .op("||")(_weighted("fts_c", "C"))
        )
        # A Core insert, not text(): the Vector column's bind processor is what
        # adapts a Python list to a pgvector `vector`, and a text() statement
        # has no such processor. The embedding bindparam is typed explicitly
        # because there is no column context for SQLAlchemy to infer it from.
        stmt = insert(Passage).values(
            note_id=bindparam("note_id"),
            client_name=bindparam("client_name"),
            ordinal=bindparam("ordinal"),
            heading_path=bindparam("heading_path"),
            content=bindparam("content"),
            embedding=bindparam("embedding", type_=Vector(settings.embedding_dimension)),
            tsv=tsv,
        )
        self.db.execute(
            stmt,
            [
                {
                    "note_id": note_id,
                    "client_name": self.client_name,  # stamped from self
                    "ordinal": passage.ordinal,
                    "heading_path": passage.heading_path,
                    "content": passage.content,
                    "embedding": passage.embedding,
                    "fts_a": passage.fts_a,
                    "fts_c": passage.fts_c,
                }
                for passage in passages
            ],
        )
        return len(passages)

    def get_note(self, note_id: int) -> Row[Any] | None:
        """One note's columns by id, or None.

        Columns rather than the ORM entity: the route serialises a fixed set
        of fields, and an entity would also populate the identity map. None
        covers both a note that does not exist and another tenant's note --
        the route cannot distinguish them, which is what makes its 404 free of
        an existence leak.
        """
        stmt = select(
            Note.id, Note.scope, Note.path_or_uri, Note.name, Note.description,
            Note.type, Note.provenance_kind, Note.provenance_date, Note.status,
            Note.content_hash, Note.pipeline_version, Note.body, Note.meta,
            Note.created_at, Note.updated_at,
        ).where(
            Note.client_name == self.client_name,
            Note.id == note_id,
        )
        return self.db.execute(stmt).one_or_none()

    def count_passages(self, note_id: int) -> int:
        """How many passages this note has.

        The tenant filter matters here too: a cross-tenant count leaks the
        existence of another tenant's passages through a number.
        """
        stmt = (
            select(func.count())
            .select_from(Passage)
            .where(
                Passage.client_name == self.client_name,
                Passage.note_id == note_id,
            )
        )
        return self.db.execute(stmt).scalar_one()

    def delete_note(self, note_id: int) -> bool:
        """Delete one note; True when a row was actually removed.

        Its passages go with it through the composite FK's ON DELETE CASCADE.
        They are deliberately NOT deleted explicitly first: a second statement
        would be a second place for the tenant filter to be wrong, and the
        cascade is exercised directly by the schema tests.

        False covers both "no such note" and "another tenant's note", for the
        same reason get_note returns None for both. Does not commit.
        """
        stmt = (
            delete(Note)
            .where(
                Note.client_name == self.client_name,
                Note.id == note_id,
            )
            .returning(Note.id)
        )
        return self.db.execute(stmt).one_or_none() is not None

    def search_fts(
        self,
        query: str,
        top_k: int,
        repo_filter: str | None = None,
    ) -> Sequence[Row[Any]]:
        """Full-text search over passages, carrying their note's metadata.

        FTS only in this slice: no vector expression and no hnsw.* setting
        anywhere, because nothing here touches the HNSW index.

        statement_timeout is set with set_config(..., true), i.e. SET LOCAL --
        scoped to this transaction, so it cannot leak onto a pooled connection
        or affect ingest and the startup schema checks. `settings` is read at
        CALL time, not captured at import, so tests can monkeypatch it.
        """
        self.db.execute(
            text("SELECT set_config('statement_timeout', :t, true)"),
            {"t": str(settings.statement_timeout_ms)},
        )

        # Read it back in a SEPARATE statement, mirroring
        # app/repositories/learning.py:182-196. The separation is the point:
        # SET LOCAL needs an open transaction block to persist past its own
        # statement, and with none open the timeout covers only the set_config
        # call itself while this search runs uncapped and still answers 200.
        # Reading it back inside the same statement would prove nothing.
        #
        # Session.in_transaction() is not the guard: it is False before the
        # first statement (autobegin is lazy) and True under AUTOCOMMIT where
        # SET LOCAL still no-ops. It inspects the Session, not the database.
        shown = self.db.execute(
            text("SELECT current_setting('statement_timeout')")
        ).scalar_one()
        if shown not in _timeout_spellings(settings.statement_timeout_ms):
            raise RuntimeError(
                f"statement_timeout setting did not take effect: expected "
                f"{settings.statement_timeout_ms}ms but got {shown!r}. "
                f"set_config(..., true) is SET LOCAL and requires an open "
                f"transaction block to persist to the next statement. With no "
                f"transaction open (e.g. an AUTOCOMMIT connection) it applies "
                f"only to its own statement and this search would run uncapped "
                f"while still answering 200."
            )

        q = func.websearch_to_tsquery("english", query)
        # Left on the default weight vector {0.1, 0.2, 0.4, 1.0} deliberately:
        # putting description at weight C depends on those defaults, so a
        # custom vector here would silently rebalance A/B/C.
        rank = func.ts_rank_cd(Passage.tsv, q).label("rank")
        stmt = (
            select(
                Passage.id, Passage.note_id, Passage.ordinal,
                Passage.heading_path, Passage.content, rank,
                Note.name, Note.description, Note.type, Note.path_or_uri,
                Note.scope, Note.provenance_kind, Note.provenance_date,
                Note.status,
            )
            .join(
                Note,
                and_(
                    Note.id == Passage.note_id,
                    Note.client_name == Passage.client_name,
                ),
            )
            # Both client_name predicates are present on purpose: the one on
            # Passage is the single-table predicate on the table the GIN index
            # scans, and the one on Note is independent of the join condition,
            # so dropping either still leaves the boundary enforced.
            .where(
                Passage.client_name == self.client_name,
                Note.client_name == self.client_name,
                Passage.tsv.op("@@")(q),
            )
            # Passage.id is a deterministic tiebreak: ts_rank_cd ties are
            # common and an unstable order makes tests flap.
            .order_by(rank.desc(), Passage.id)
            .limit(top_k)
        )
        if repo_filter:
            stmt = stmt.where(Note.scope == repo_filter)

        return self.db.execute(stmt).all()


def _timeout_spellings(milliseconds: int) -> frozenset[str]:
    """Every rendering Postgres may give back for this many milliseconds.

    `current_setting('statement_timeout')` normalises an integer GUC to the
    largest unit that divides it evenly -- 5000 comes back as "5s", 60000 as
    "1min", 4321 stays "4321ms", 0 is a bare "0". So the read-back guard cannot
    compare against f"{ms}ms" alone: on the default 5000 that spelling never
    matches and the guard would fire on a correctly applied setting.
    """
    if milliseconds == 0:
        return frozenset({"0", "0ms"})
    spellings = {f"{milliseconds}ms"}
    for unit, size in (("s", 1_000), ("min", 60_000), ("h", 3_600_000)):
        if milliseconds % size == 0:
            spellings.add(f"{milliseconds // size}{unit}")
    return frozenset(spellings)
