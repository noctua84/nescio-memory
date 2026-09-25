from pathlib import Path

from fastapi import APIRouter, Depends, Form
from langfuse import observe
from sqlalchemy.orm import Session

from app.core.chunking import chunk_text
from app.core.db import get_db
from app.core.embeddings import get_embedding
from app.models.learning import Learning
from app.repositories import LearningRepository
from app.schemas.ingest import IngestResponse

router = APIRouter()


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
):
    repo = LearningRepository(db)
    repo.delete_by_file(repo_name, file_path)

    ingested = 0
    for i, chunk in enumerate(chunk_text(content)):
        if len(chunk.strip()) < 50:
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