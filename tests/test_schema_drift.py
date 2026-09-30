"""The model and the migrations must not drift apart.

Real migrations only reveal drift that breaks an exercised statement. A missing
index breaks nothing, which is exactly how learnings.file_path went unindexed
while every test passed. This closes that gap.
"""
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from app.models import Base

# Divergences that are expected and accepted. Each entry is (operation, object
# name). Anything not listed here is a regression.
ACCEPTED_DRIFT = {
    # PostgreSQL has no practical distinction between TEXT and VARCHAR without a
    # length, so this reflects differently without meaning anything.
    ("modify_type", "api_keys.client_name"),
    # The HNSW index is created by migration and cannot be expressed in the
    # model, so autogenerate always reports it as removable. Deleting it would
    # destroy vector search performance.
    ("remove_index", "learnings_embedding_idx"),
}


def _describe(diff) -> tuple[str, str]:
    """Reduce an autogenerate diff entry to (operation, object name)."""
    if isinstance(diff, list):
        # A column alteration arrives as a list of tuples.
        diff = diff[0]
    operation = diff[0]
    if operation.endswith("_index") or operation.endswith("_constraint"):
        return operation, diff[1].name
    if operation.endswith("_table"):
        return operation, diff[1].name
    if operation.endswith("_column"):
        return operation, f"{diff[2]}.{diff[3].name}"
    # modify_* entries carry the table and column in fixed positions.
    return operation, f"{diff[2]}.{diff[3]}"


def test_the_models_and_the_migrated_schema_agree(engine):
    with engine.connect() as connection:
        diffs = compare_metadata(
            MigrationContext.configure(connection), Base.metadata
        )

    unexpected = [
        described
        for described in (_describe(diff) for diff in diffs)
        if described not in ACCEPTED_DRIFT
    ]

    assert not unexpected, (
        "model and migrations have drifted: "
        f"{unexpected}. If a divergence is genuinely intended, add it to "
        "ACCEPTED_DRIFT with a comment explaining why."
    )
