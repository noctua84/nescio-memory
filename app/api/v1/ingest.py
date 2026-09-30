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
                f"{MAX_CONTENT_CHARS} limit"
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