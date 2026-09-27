"""Proves the harness itself, before any application behaviour is asserted.

If these fail, nothing else in the suite can be trusted.
"""
from sqlalchemy import inspect, select, text

from app.models.learning import Learning


def test_migrations_created_every_table(engine):
    tables = set(inspect(engine).get_table_names())
    assert {"learnings", "api_keys", "alembic_version"} <= tables


def test_the_vector_extension_is_installed(engine):
    with engine.connect() as connection:
        installed = connection.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
    assert installed == 1


def test_the_embedding_column_has_the_configured_dimension(engine):
    # Migration 0002 moved this column to 384 dimensions. If create_all() were
    # ever substituted for real migrations this assertion would still pass,
    # which is why the HNSW check below exists too.
    with engine.connect() as connection:
        dimension = connection.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'learnings'::regclass AND attname = 'embedding'"
            )
        ).scalar()
    assert dimension == 384


def test_the_hnsw_index_exists(engine):
    # This index is created only by migration 8e6b572b0ae3 and rebuilt by 0002.
    # It is absent from the SQLAlchemy model, so its presence is proof that
    # real migrations ran rather than metadata.create_all().
    with engine.connect() as connection:
        indexes = connection.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'learnings'")
        ).scalars().all()
    assert "learnings_embedding_idx" in indexes


def _probe_row(client_name: str) -> Learning:
    return Learning(
        client_name=client_name,
        repo_name="repo_a",
        file_path="probe.md",
        content="written by an isolation probe",
        meta={"chunk_index": 0},
        embedding=[0.0] * 384,
    )


def test_a_row_written_in_a_test_is_visible_within_that_test(db_session):
    db_session.add(_probe_row("visibility_probe"))
    db_session.flush()
    found = db_session.scalars(
        select(Learning).where(Learning.client_name == "visibility_probe")
    ).all()
    assert len(found) == 1


def test_an_application_level_commit_does_not_escape_the_transaction(
    db_session, engine
):
    """The ingest endpoint calls db.commit() itself.

    Under join_transaction_mode="create_savepoint" that releases a savepoint
    instead of committing the outer transaction. Proven here by reading through
    a second, independent connection while the test transaction is still open:
    the row must be visible to this test own session and invisible to everyone
    else.

    Deliberately self-contained rather than split across two ordered tests. It
    cannot pass vacuously: the first assertion fails if the row was never
    written, so the second is only ever reached with a real row in play.
    """
    db_session.add(_probe_row("commit_probe"))
    db_session.commit()

    assert db_session.scalars(
        select(Learning).where(Learning.client_name == "commit_probe")
    ).all(), "the row is not visible to the session that wrote it"

    with engine.connect() as observer:
        escaped = observer.execute(
            select(Learning.id).where(Learning.client_name == "commit_probe")
        ).all()
    assert escaped == [], "a committed row escaped the test transaction"
