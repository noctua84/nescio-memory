"""One client's limits do not touch another's, and one client's keys share theirs."""
import pytest

from app.config import settings
from tests._rate_limit_helpers import (
    clear_of_window_boundary,
    held_slots,
    ingest,
    search,
)
from tests.factories import make_api_key

BODY = "The marmoset throughput ledger for the quarter, with no headings at all."
HOUR = 3600


@pytest.fixture
def two_per_hour(engine, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_search_requests", 2)
    monkeypatch.setattr(settings, "rate_limit_window_s", HOUR)
    clear_of_window_boundary(engine, HOUR)


def test_another_clients_ingest_is_unaffected_by_held_slots(client, db_session, engine):
    acme = make_api_key(db_session, "acme")
    globex = make_api_key(db_session, "globex")

    with held_slots(engine, "acme", 0, 1):
        assert ingest(client, acme, BODY).status_code == 429
        assert ingest(client, globex, BODY).status_code == 200


def test_another_clients_search_is_unaffected_by_an_exhausted_budget(
    client, db_session, two_per_hour
):
    acme = make_api_key(db_session, "acme")
    globex = make_api_key(db_session, "globex")

    assert [search(client, acme).status_code for _ in range(3)] == [200, 200, 429]

    assert [search(client, globex).status_code for _ in range(2)] == [200, 200]


def test_the_overlapping_keys_of_a_rotation_share_one_search_budget(
    client, db_session, two_per_hour
):
    old = make_api_key(db_session, "acme", plaintext="nm_test_acme_old")
    new = make_api_key(db_session, "acme", plaintext="nm_test_acme_new")

    assert search(client, old).status_code == 200
    assert search(client, new).status_code == 200

    assert search(client, old).status_code == 429
    assert search(client, new).status_code == 429


def test_the_overlapping_keys_of_a_rotation_share_their_ingest_slots(
    client, db_session, engine
):
    """Slots are held by client_name, so a second key finds them just as taken."""
    old = make_api_key(db_session, "acme", plaintext="nm_test_acme_old")
    new = make_api_key(db_session, "acme", plaintext="nm_test_acme_new")

    with held_slots(engine, "acme", 0, 1):
        assert ingest(client, old, BODY).status_code == 429
        assert ingest(client, new, BODY).status_code == 429
