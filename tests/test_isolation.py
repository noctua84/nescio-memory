"""The tenant boundary.

Every failure mode here is silent in production: no exception, no error
response, just one client reading or destroying another client's data.
"""
from sqlalchemy import select

from app.models.learning import Learning
from tests.factories import make_api_key, make_learning

LONG_ENOUGH = (
    "This content comfortably exceeds the fifty character minimum that the "
    "ingest endpoint applies to each chunk."
)


def test_search_does_not_return_another_clients_learnings(client, db_session):
    make_learning(db_session, "acme", content="a private acme note")
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/search",
        json={"query": "a private acme note", "top_k": 50},
        headers={"X-API-Key": globex_key},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_a_client_sees_only_its_own_rows_when_both_exist(client, db_session):
    make_learning(db_session, "acme", content="acme flavoured content")
    make_learning(db_session, "globex", content="globex flavoured content")
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/search",
        json={"query": "flavoured content", "top_k": 50},
        headers={"X-API-Key": globex_key},
    )

    contents = [result["content"] for result in response.json()["results"]]
    assert contents == ["globex flavoured content"]


def test_reingest_by_another_client_does_not_delete_existing_rows(client, db_session):
    # delete_by_file filters on client_name. Without that filter, this ingest
    # would wipe acme's copy of the same path.
    make_learning(
        db_session,
        "acme",
        repo_name="shared",
        file_path="docs/note.md",
        content="acme content for the shared path",
    )
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/ingest",
        data={
            "repo_name": "shared",
            "file_path": "docs/note.md",
            "content": LONG_ENOUGH,
        },
        headers={"X-API-Key": globex_key},
    )
    assert response.status_code == 200

    db_session.expire_all()
    acme_rows = db_session.scalars(
        select(Learning).where(Learning.client_name == "acme")
    ).all()
    assert len(acme_rows) == 1
    assert acme_rows[0].content == "acme content for the shared path"


def test_the_same_path_coexists_under_two_clients(client, db_session):
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    payload = {
        "repo_name": "shared",
        "file_path": "docs/note.md",
        "content": LONG_ENOUGH,
    }

    for key in (acme_key, globex_key):
        assert (
            client.post(
                "/api/v1/ingest", data=payload, headers={"X-API-Key": key}
            ).status_code
            == 200
        )

    db_session.expire_all()
    owners = db_session.scalars(
        select(Learning.client_name).where(Learning.file_path == "docs/note.md")
    ).all()
    assert sorted(set(owners)) == ["acme", "globex"]


def test_the_client_filter_is_what_excludes_the_row(client, db_session):
    """Two requests differing only in the key presented; one sees the row.

    This is the teeth of the isolation suite. Because the requests are
    otherwise identical, a passing pair cannot be explained by anything except
    the client_name filter, so no mutation of production code is needed to show
    that the filter is load-bearing.
    """
    make_learning(db_session, "acme", content="the contested row")
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    payload = {"query": "the contested row", "top_k": 50}

    owner_view = client.post(
        "/api/v1/search", json=payload, headers={"X-API-Key": acme_key}
    )
    other_view = client.post(
        "/api/v1/search", json=payload, headers={"X-API-Key": globex_key}
    )

    assert [r["content"] for r in owner_view.json()["results"]] == [
        "the contested row"
    ]
    assert other_view.json()["results"] == []


def test_ingest_stamps_the_authenticated_clients_name(client, db_session):
    # LearningRepository.add stamps client_name; the endpoint never passes it.
    key = make_api_key(db_session, "acme")

    client.post(
        "/api/v1/ingest",
        data={
            "repo_name": "repo_a",
            "file_path": "docs/note.md",
            "content": LONG_ENOUGH,
        },
        headers={"X-API-Key": key},
    )

    db_session.expire_all()
    owners = db_session.scalars(select(Learning.client_name)).all()
    assert owners
    assert set(owners) == {"acme"}
