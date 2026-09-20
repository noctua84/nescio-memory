import json
from pathlib import Path

from fastapi import APIRouter, Form
from langfuse import observe

from app.core.chunking import chunk_text
from app.core.db import get_db_connection
from app.core.embeddings import get_embedding
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
):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM learnings WHERE repo_name = %s AND file_path = %s",
                (repo_name, file_path),
            )

            ingested = 0
            for i, chunk in enumerate(chunk_text(content)):
                if len(chunk.strip()) < 50:
                    continue

                metadata = {
                    "file_name": Path(file_path).name,
                    "relative_path": file_path,
                    "chunk_index": i,
                }
                cur.execute(
                    """
                    INSERT INTO learnings
                        (repo_name, file_path, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (repo_name, file_path, chunk, json.dumps(metadata),
                     get_embedding(chunk)),
                )
                ingested += 1

        conn.commit()
    finally:
        conn.close()

    return IngestResponse(
        status="success", file_path=file_path, chunks_ingested=ingested
    )