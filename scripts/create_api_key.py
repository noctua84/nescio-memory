# scripts/create_api_key.py
import sys

from app.core.db import get_sessionmaker
from app.core.security import generate_api_key
from app.models.api_key import ApiKey


def main() -> None:
    client_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    plaintext, key_hash, prefix = generate_api_key(client_name)

    with get_sessionmaker()() as db:
        db.add(ApiKey(key_prefix=prefix, key_hash=key_hash, client_name=client_name))
        db.commit()

    print(f"client_name : {client_name}")
    print(f"api_key     : {plaintext}")
    print("⚠️  Store this now — it cannot be retrieved again.")


if __name__ == "__main__":
    main()