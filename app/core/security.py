# app/core/security.py
import hashlib
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.api_key import ApiKey

API_KEY_HEADER = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _hash_key(key: str) -> str:
    # API keys are high-entropy, so a fast hash is appropriate (unlike passwords).
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key(client_name: str) -> tuple[str, str, str]:
    """Returns (plaintext_key, key_hash, key_prefix). Show plaintext ONCE."""
    plaintext = f"nm_{secrets.token_urlsafe(32)}"
    return plaintext, _hash_key(plaintext), plaintext[:15]


def get_current_client(
    api_key: str | None = Security(_api_key_header),
    db: Session = Depends(get_db),
) -> ApiKey:
    if not api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing API key")

    stmt = select(ApiKey).where(
        ApiKey.key_hash == _hash_key(api_key),
        ApiKey.revoked_at.is_(None),
    )
    record = db.execute(stmt).scalar_one_or_none()
    if record is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or revoked API key")
    return record