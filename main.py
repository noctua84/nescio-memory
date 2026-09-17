import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Form, HTTPException
from langfuse import observe
import psycopg2
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel

from config import settings

""" Helper functions """
def get_db_connection():
    conn = psycopg2.connect(settings.database_url)
    register_vector(conn)
    return conn

def get_embedding(text: str) -> list[float]:
    payload = { "model": settings.ollama_model, "prompt": text }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(settings.ollama_url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

""" Models """
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    repo_filter: str | None = None

class SearchResponse(BaseModel):
    results: list[dict]

class IngestResponse(BaseModel):
    status: str
    file_path: str
    chunks_ingested: int

""" Endpoints """
app = FastAPI(title=settings.app_name)
@app.post("/search", response_model=SearchResponse)
@observe(name="semantic-search")
def search_memory(request: SearchRequest):
    query_embedding = get_embedding(request.query)

    conn = get_db_connection()
    cur = conn.cursor()

    sql = """
        SELECT content, metadata, 1 - (embedding <=> %s) as similarity
        FROM learnings
    """

    params: list[Any] = [query_embedding]

    if request.repo_filter:
        sql += " WHERE metadata->>'repo_name' = %s"
        params.append(request.repo_filter)

    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params.extend([query_embedding, request.top_k])

    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = [
        {"content": row[0], "metadata": row[1], "similarity": float(row[2])} for row in rows
    ]

    return SearchResponse(results=results)


@app.post("/ingest", response_model=IngestResponse)
@observe(name="file-ingestion")  # Automatically traces the ingestion process
def ingest_file(
        repo_name: str = Form(...),
        file_path: str = Form(...),
        content: str = Form(...)
):
    conn = get_db_connection()
    cur = conn.cursor()

    # Delete old chunks for this specific file to handle updates/overwrites
    cur.execute(
        "DELETE FROM learnings WHERE repo_name = %s AND file_path = %s",
        (repo_name, file_path)
    )

    chunks = chunk_text(content)
    ingested_count = 0

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue

        embedding = get_embedding(chunk)

        metadata = {
            "file_name": Path(file_path).name,
            "relative_path": file_path,
            "chunk_index": i
        }

        cur.execute("""
                    INSERT INTO learnings (repo_name, file_path, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """, (repo_name, file_path, chunk, json.dumps(metadata), embedding))
        ingested_count += 1

    conn.commit()
    cur.close()
    conn.close()

    return IngestResponse(
        status="success",
        file_path=file_path,
        chunks_ingested=ingested_count
    )


@app.get("/health")
def health():
    return {"status": "ok", "ollama_model": settings.ollama_model}