# app/core/security.py
import hashlib
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.key_usage import KeyUsageRecorder, get_key_usage_recorder, key_use_is_due
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
    record_use: KeyUsageRecorder = Depends(get_key_usage_recorder),
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
    # Everything the rest of the request reads from `record` is read here, while
    # the transaction is open. The invariant in get_db's docstring ends it before
    # this dependency returns: the next dependency and the route are separate
    # threadpool tasks, and a session left in transaction would keep its pooled
    # connection while queued for a thread.
    #
    # expunge() before commit(): commit would expire `record`, and the first read
    # of client_name or id in a later task would then re-SELECT, re-opening the
    # very transaction just released. Detached, it keeps the columns it loaded;
    # ApiKey has no relationships, so nothing needs the session again.
    key_id = record.id
    touch_due = key_use_is_due(record.last_used_at)
    db.expunge(record)
    db.commit()
    # After the commit, so the touch takes a connection while this request holds
    # none. Inside the transaction it would nest a second checkout under the
    # first, and a burst the size of the pool could hold every connection while
    # each waits for another.
    if touch_due:
        record_use(key_id)
    return record