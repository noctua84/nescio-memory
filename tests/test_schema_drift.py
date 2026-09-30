"""The model and the migrations must not drift apart.

Real migrations only reveal drift that breaks an exercised statement. A missing
index breaks nothing, which is exactly how learnings.file_path went unindexed
while every test passed. This closes that gap.
"""
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from app.models import Base

# No accepted divergences. The model describes the schema exactly, including the
# HNSW index, which SQLAlchemy's PostgreSQL dialect can express. Adding an entry
# here should be a deliberate, justified act -- an allowlisted object is one this
# test stops guarding, and a divergence on it goes unreported.
ACCEPTED_DRIFT: set[tuple[str, str]] = set()


def _describe(diff) -> list[tuple[str, str]]:
    """Reduce an autogenerate diff entry to (operation, object name) pairs.

    Returns a list because Alembic packs every alteration to a single column into
    one list entry. Describing only the first would hide the rest.
    """
    if isinstance(diff, list):
        return [described for entry in diff for described in _describe(entry)]

    operation = diff[0]
    if operation.endswith(("_index", "_constraint", "_fk", "_table")):
        return [(operation, diff[1].name)]
    if operation.endswith("_column"):
        return [(operation, f"{diff[2]}.{diff[3].name}")]
    if operation.startswith("modify_"):
        return [(operation, f"{diff[2]}.{diff[3]}")]
    raise AssertionError(
        f"unrecognised autogenerate diff shape: {diff!r}. Teach _describe about "
        "it rather than letting it fall through silently."
    )


def test_the_models_and_the_migrated_schema_agree(engine):
    with engine.connect() as connection:
        diffs = compare_metadata(
            MigrationContext.configure(connection), Base.metadata
        )

    unexpected = [
        described
        for diff in diffs
        for described in _describe(diff)
        if described not in ACCEPTED_DRIFT
    ]

    assert not unexpected, (
        "model and migrations have drifted: "
        f"{unexpected}. If a divergence is genuinely intended, add it to "
        "ACCEPTED_DRIFT with a comment explaining why."
    )
