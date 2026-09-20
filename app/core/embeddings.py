import httpx

from app.config import settings


def get_embedding(text: str) -> list[float]:
    payload = {"model": settings.ollama_model, "prompt": text}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(settings.ollama_url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]