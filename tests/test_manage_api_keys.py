"""scripts/manage_api_keys.py and its create_api_key.py shim, run in-process.

Every test here runs against the real Postgres container through the same
``db_session`` the rest of the integration suite uses, and the ``client``
fixture authenticates against that same session. A revoke made by the script
is therefore visible to authentication exactly as it would be in production.
Nothing is committed: the script's commit releases a savepoint inside the
test's outer transaction, which the ``connection`` fixture rolls back.
"""
import re
import secrets
from contextlib import nullcontext

import pytest
from sqlalchemy import select

from app.core.security import _hash_key
from app.models.api_key import ApiKey
from scripts import create_api_key, manage_api_keys
from tests.factories import make_api_key

SEARCH_PAYLOAD = {"q": "anything", "top_k": 5}


def _secret() -> str:
    """A high-entropy plaintext in the production shape, so the prefix and
    the remainder can be told apart in the assertions below."""
    return f"nm_{secrets.token_urlsafe(32)}"


def _run(db_session, argv):
    return manage_api_keys.main(argv, session_factory=lambda: nullcontext(db_session))


def _id_of(db_session, plaintext: str) -> int:
    stmt = select(ApiKey.id).where(ApiKey.key_hash == _hash_key(plaintext))
    return db_session.execute(stmt).scalar_one()


def _authenticates(client, plaintext: str) -> bool:
    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": plaintext}
    )
    if response.status_code == 401:
        return False
    assert response.status_code == 200, response.text
    return True


def test_list_shows_active_keys_and_never_a_secret(db_session, capsys):
    acme, globex = _secret(), _secret()
    make_api_key(db_session, "acme", plaintext=acme)
    make_api_key(db_session, "globex", plaintext=globex)

    assert _run(db_session, ["list"]) == 0
    out = capsys.readouterr().out

    assert "acme" in out
    assert "globex" in out
    for secret in (acme, globex):
        # The whole key, its tail beyond the display prefix, and its hash
        # must all be absent. Only key_prefix (the first 15 characters) is
        # shown, by design, for telling keys apart.
        assert secret not in out
        assert secret[15:] not in out
        assert _hash_key(secret) not in out


def test_list_hides_revoked_keys_unless_asked(db_session, capsys):
    make_api_key(db_session, "retired", plaintext=_secret(), revoked=True)
    make_api_key(db_session, "active", plaintext=_secret())

    assert _run(db_session, ["list"]) == 0
    default_out = capsys.readouterr().out
    assert "active" in default_out
    assert "retired" not in default_out

    assert _run(db_session, ["list", "--include-revoked"]) == 0
    full_out = capsys.readouterr().out
    assert "retired" in full_out
    assert "active" in full_out


def test_list_filters_to_one_client(db_session, capsys):
    make_api_key(db_session, "acme", plaintext=_secret())
    make_api_key(db_session, "globex", plaintext=_secret())

    assert _run(db_session, ["list", "--client", "globex"]) == 0
    out = capsys.readouterr().out

    assert "globex" in out
    assert "acme" not in out


def test_revoke_by_id_makes_the_key_fail_authentication(db_session, client, capsys):
    key = _secret()
    make_api_key(db_session, "acme", plaintext=key)
    assert _authenticates(client, key)

    # The request above can log through the app's stdout handler; drain it so
    # the assertion below sees only the script's own output.
    capsys.readouterr()

    assert _run(db_session, ["revoke", "--id", str(_id_of(db_session, key))]) == 0
    assert capsys.readouterr().out.strip() == "revoked 1"

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or revoked API key"}


def test_revoke_by_client_revokes_only_that_clients_keys(db_session, client, capsys):
    acme, globex = _secret(), _secret()
    make_api_key(db_session, "acme", plaintext=acme)
    make_api_key(db_session, "globex", plaintext=globex)

    assert _run(db_session, ["revoke", "--client", "acme", "--yes"]) == 0
    assert capsys.readouterr().out.strip() == "revoked 1"

    assert not _authenticates(client, acme)
    assert _authenticates(client, globex)


def test_revoking_an_already_revoked_key_reports_zero(db_session, capsys):
    key = _secret()
    make_api_key(db_session, "acme", plaintext=key)
    key_id = str(_id_of(db_session, key))

    assert _run(db_session, ["revoke", "--id", key_id]) == 0
    assert capsys.readouterr().out.strip() == "revoked 1"

    assert _run(db_session, ["revoke", "--id", key_id]) == 0
    assert capsys.readouterr().out.strip() == "revoked 0"


def test_revoke_by_a_missing_id_exits_non_zero(db_session, capsys):
    assert _run(db_session, ["revoke", "--id", "999999999"]) == 1
    captured = capsys.readouterr()
    assert "999999999" in captured.err
    assert captured.out == ""


def test_revoke_by_client_without_yes_refuses_and_changes_nothing(db_session, client, capsys):
    key = _secret()
    make_api_key(db_session, "acme", plaintext=key)

    assert _run(db_session, ["revoke", "--client", "acme"]) == 1
    assert "--yes" in capsys.readouterr().err

    assert _authenticates(client, key)


def test_revoke_needs_exactly_one_of_id_or_client(db_session):
    with pytest.raises(SystemExit):
        _run(db_session, ["revoke"])
    with pytest.raises(SystemExit):
        _run(db_session, ["revoke", "--id", "1", "--client", "acme"])


def test_create_prints_a_key_that_authenticates_once(db_session, client, capsys):
    assert _run(db_session, ["create", "acme"]) == 0
    out = capsys.readouterr().out

    assert "client_name : acme" in out
    assert "Store this now" in out
    key = re.search(r"api_key\s*:\s*(\S+)", out).group(1)
    assert key.startswith("nm_")
    assert _authenticates(client, key)


def test_the_create_api_key_shim_still_mints_a_usable_key(db_session, client, capsys):
    rc = create_api_key.main(["acme"], session_factory=lambda: nullcontext(db_session))
    assert rc == 0
    out = capsys.readouterr().out

    assert "client_name : acme" in out
    key = re.search(r"api_key\s*:\s*(\S+)", out).group(1)
    assert _authenticates(client, key)


def test_the_create_api_key_shim_defaults_the_client_name(db_session, capsys):
    rc = create_api_key.main([], session_factory=lambda: nullcontext(db_session))
    assert rc == 0
    assert "client_name : default" in capsys.readouterr().out
