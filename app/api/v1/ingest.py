from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException
from langfuse import observe
from sqlalchemy.orm import Session

from app.core.chunking import chunk_text
from app.core.db import get_db
from app.core.embeddings import get_embedding
from app.core.security import get_current_client
from app.models import ApiKey
from app.models.learning import Learning
from app.repositories.learning import LearningRepository
from app.schemas.ingest import IngestResponse

router = APIRouter()

MAX_CONTENT_CHARS = 500_000   # ~500 KB per file
MIN_CONTENT_CHARS = 50

# Starlette's form parser enforces this per-field BYTE limit
# (starlette/formparsers.py: FormParser/MultiPartParser default
# max_part_size = 1024 * 1024) before this module's Form(...) fields are ever
# bound to a request, let alone before _validate_content_size below runs.
# FastAPI's routing (fastapi/routing.py: `body = await request.form()`) calls
# Request.form() with no arguments, so there is no application- or
# route-level hook to raise it for a Form(...)-declared route without
# bypassing Form(...) entirely and hand-parsing the body -- which would
# change this endpoint's documented request schema. So for multi-byte
# content this byte limit, not MAX_CONTENT_CHARS, is the one that actually
# fires, and it does so via Starlette's own generic message rather than
# ours. See docs/superpowers/specs/2026-09-29-stabilization-design.md item 2
# review and .superpowers/sdd/task-robustness-report.md for the investigation.
FORM_PART_BYTE_LIMIT = 1024 * 1024


def _validate_file_path(file_path: str) -> None:
    """Reject absolute paths and traversal — file_path is stored/reflected."""
    if file_path.startswith("/") or ".." in Path(file_path).parts:
        raise HTTPException(
            status_code=400,
            detail="file_path must be a relative path without '..'",
        )


def _validate_content_size(content: str) -> None:
    """Reject content above the cap before anything is deleted or embedded."""
    if len(content) > MAX_CONTENT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"content is too large: {len(content)} characters exceeds the "
                f"{MAX_CONTENT_CHARS} limit. Note: a {FORM_PART_BYTE_LIMIT}-byte "
                "per-field limit is also enforced ahead of this check, which "
                "multi-byte (non-ASCII) content can reach at a much lower "
                "character count."
            ),
        )

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Markdown File",
    description="Chunks content, embeds via Ollama, and upserts into pgvector.",
)
@observe(name="file-ingestion")
def ingest_file(
    repo_name: str = Form(..., description="Repository this file belongs to."),
    file_path: str = Form(..., description="Relative path of the file."),
    content: str = Form(..., description="Raw markdown content to vectorize."),
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client)
):
    _validate_file_path(file_path)
    _validate_content_size(content)
    repo = LearningRepository(db, client_name=client.client_name)
    repo.delete_by_file(repo_name, file_path)

    ingested = 0
    for i, chunk in enumerate(chunk_text(content)):
        if len(chunk.strip()) < MIN_CONTENT_CHARS:
            continue

        repo.add(
            Learning(
                repo_name=repo_name,
                file_path=file_path,
                content=chunk,
                meta={
                    "file_name": Path(file_path).name,
                    "relative_path": file_path,
                    "chunk_index": i,
                },
                embedding=get_embedding(chunk),
            )
        )
        ingested += 1

    db.commit()  # single transaction boundary for the whole file
    return IngestResponse(status="success", file_path=file_path, chunks_ingested=ingested)