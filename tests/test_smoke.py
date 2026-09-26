"""Proves the harness itself, before any application behaviour is asserted.

If these fail, nothing else in the suite can be trusted.
"""
from sqlalchemy import inspect, text


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
