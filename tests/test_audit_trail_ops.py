"""The audit trail's surroundings: the README's query recipes, the prune script,
Langfuse attribution and the settings.

The other half of tests/test_audit_trail.py, split out to keep each module under
the length the project asks for; the small helpers are repeated rather than shared,
as tests/test_ingest_validation.py does.
"""
import hashlib
import json
import re
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from sqlalchemy import insert, select, text

from app.api.v1 import ingest as ingest_module
from app.api.v1 import notes as notes_module
from app.api.v1 import search as search_module
from app.config import Settings, settings
from app.core import tracing as tracing_module
from app.core.pipeline import content_hash
from app.core.security import _hash_key
from app.main import app
from app.models.api_key import ApiKey
from app.models.ingest_event import IngestEvent
from scripts import prune_ingest_events
from tests.factories import make_api_key, make_note, make_passage

README = Path(__file__).resolve().parent.parent / "README.md"

PATH = "people/ada-lovelace.md"
PATH_HASH = hashlib.sha256(PATH.encode("utf-8")).hexdigest()
SCOPE = "repo_a"
BODY = "The marmoset throughput ledger for the quarter, with no headings at all."
# Long sections: the splitter merges short ones, and a replace has to change the
# passage count for its before and after to be told apart.
NEW_BODY = (
    "# One\n\n" + "First rewritten section. " * 80
    + "\n\n# Two\n\n" + "Second rewritten section. " * 80 + "\n"
)


def _auth(key):
    return {"X-API-Key": key}


def _ingest(client, key, content=BODY, **overrides):
    payload = {"scope": SCOPE, "path_or_uri": PATH, "content": content, **overrides}
    return client.post("/api/v1/ingest", json=payload, headers=_auth(key))


def _events(db_session):
    db_session.expire_all()
    return db_session.execute(select(IngestEvent).order_by(IngestEvent.id)).scalars().all()


def _key_id(db_session, plaintext):
    return db_session.execute(
        select(ApiKey.id).where(ApiKey.key_hash == _hash_key(plaintext))
    ).scalar_one()


def _seed(db_session, passages=2):
    note = make_note(db_session, "acme", scope=SCOPE, path_or_uri=PATH)
    for ordinal in range(passages):
        make_passage(db_session, note, ordinal=ordinal, content=f"{PATH} {ordinal}")
    return note


# --- Operator recipes ---------------------------------------------------------


def _readme_recipe(name):
    """The SQL under a README recipe marker, so the documented query is the one
    that is run here rather than a copy that can drift from it."""
    text_ = README.read_text(encoding="utf-8")
    match = re.search(rf"<!-- recipe: {name} -->\s*```sql\n(.*?)```", text_, flags=re.DOTALL)
    assert match, f"no recipe {name!r} in the README"
    return match.group(1)


def test_the_readme_recipe_answers_which_key_replaced_note_x_on_date_d(client, db_session):
    other_key = make_api_key(db_session, "acme", plaintext="nm_test_acme_other")
    key = make_api_key(db_session, "acme")
    first = _ingest(client, other_key).json()
    second = _ingest(client, key, content=NEW_BODY).json()
    today = db_session.execute(text("SELECT CAST(now() AS date)")).scalar_one()

    rows = db_session.execute(
        text(_readme_recipe("replaced-on-date")),
        {"client_name": "acme", "note_id": first["note_id"], "day": today},
    ).all()

    assert len(rows) == 1
    row = rows[0]
    assert row.api_key_id == _key_id(db_session, key)
    assert row.key_prefix == key[:15]
    assert row.passages_before == first["passages"]
    assert row.passages_after == second["passages"]
    assert row.content_hash_before == content_hash(BODY)
    assert row.content_hash_after == content_hash(NEW_BODY)

    # Another day answers nothing, so the date bounds really are applied.
    elsewhere = db_session.execute(
        text(_readme_recipe("replaced-on-date")),
        {
            "client_name": "acme",
            "note_id": first["note_id"],
            "day": today - timedelta(days=2),
        },
    ).all()
    assert elsewhere == []


def test_the_readme_recipe_lists_everything_that_happened_to_a_path(client, db_session):
    key = make_api_key(db_session, "acme")
    note_id = _ingest(client, key).json()["note_id"]
    _ingest(client, key, content=NEW_BODY)
    client.delete(f"/api/v1/notes/{note_id}", headers=_auth(key))

    rows = db_session.execute(
        text(_readme_recipe("path-history")),
        {"client_name": "acme", "path_sha256": PATH_HASH},
    ).all()

    assert [row.action for row in rows] == ["created", "replaced", "deleted"]


def test_the_readme_recipe_lists_what_one_key_did(client, db_session):
    key = make_api_key(db_session, "acme")
    other = make_api_key(db_session, "acme", plaintext="nm_test_acme_other")
    _ingest(client, key)
    _ingest(client, other, path_or_uri="docs/other.md")

    rows = db_session.execute(
        text(_readme_recipe("key-history")), {"api_key_id": _key_id(db_session, key)}
    ).all()

    assert [row.path_or_uri for row in rows] == [PATH]


# --- Prune --------------------------------------------------------------------


def _event_aged(db_session, days):
    return db_session.execute(
        insert(IngestEvent)
        .values(
            at=datetime.now(UTC) - timedelta(days=days),
            client_name="acme",
            api_key_id=1,
            scope=SCOPE,
            note_id=days,
            action="deleted",
            path_sha256=PATH_HASH,
        )
        .returning(IngestEvent.note_id)
    ).scalar_one()


def _prune(db_session):
    return prune_ingest_events.main([], session_factory=lambda: nullcontext(db_session))


def test_prune_deletes_only_rows_past_the_retention_period(db_session, capsys, monkeypatch):
    monkeypatch.setattr(settings, "audit_retention_days", 100)
    for days in (1, 99, 101, 500):
        _event_aged(db_session, days)

    assert _prune(db_session) == 0

    assert sorted(event.note_id for event in _events(db_session)) == [1, 99]
    assert capsys.readouterr().out.strip() == "pruned 2 ingest_events rows older than 100 days"


def test_prune_uses_the_default_retention(db_session, capsys):
    assert settings.audit_retention_days == 400
    _event_aged(db_session, 399)
    _event_aged(db_session, 401)

    assert _prune(db_session) == 0

    assert [event.note_id for event in _events(db_session)] == [399]


def test_prune_with_nothing_to_do_reports_zero(db_session, capsys):
    assert _prune(db_session) == 0
    assert capsys.readouterr().out.strip().startswith("pruned 0 ")


# --- Langfuse attribution -----------------------------------------------------


class _Tracing:
    """Records what the routes hand Langfuse. Stands in for both entry points in
    each route module, since each binds its own names."""

    def __init__(self):
        self.propagated = []
        self.span_updates = []

    def propagate_attributes(self, **attributes):
        self.propagated.append(attributes)
        return nullcontext()

    def annotate_span(self, **metadata):
        self.span_updates.append({"metadata": metadata})

    def everything(self):
        return json.dumps([self.propagated, self.span_updates], default=str)


@pytest.fixture
def tracing(monkeypatch):
    recorder = _Tracing()
    for module in (ingest_module, notes_module, search_module):
        monkeypatch.setattr(module, "propagate_attributes", recorder.propagate_attributes)
        monkeypatch.setattr(module, "annotate_span", recorder.annotate_span)
    return recorder


def test_ingest_attributes_its_trace_and_never_the_path(client, db_session, tracing):
    key = make_api_key(db_session, "acme")

    body = _ingest(client, key).json()

    assert tracing.propagated == [{"user_id": "acme", "metadata": {"scope": SCOPE}}]
    assert tracing.span_updates == [
        {
            "metadata": {
                "note_id": body["note_id"],
                "skipped": False,
                "passages": body["passages"],
            }
        }
    ]
    assert PATH not in tracing.everything()
    assert "ada-lovelace" not in tracing.everything()


def test_a_skipped_ingest_says_so_in_its_trace(client, db_session, tracing):
    key = make_api_key(db_session, "acme")
    _ingest(client, key)

    _ingest(client, key)

    assert tracing.span_updates[-1]["metadata"]["skipped"] is True


@pytest.mark.parametrize(
    ("method_path", "params"),
    [
        pytest.param("id", None, id="by-id"),
        pytest.param("/api/v1/notes", {"scope": SCOPE, "path_or_uri": PATH}, id="by-path"),
        pytest.param("/api/v1/scopes", {"scope": SCOPE}, id="by-scope"),
    ],
)
def test_every_delete_attributes_its_trace_and_never_the_path(
    client, db_session, tracing, method_path, params
):
    key = make_api_key(db_session, "acme")
    note = _seed(db_session, passages=2)
    note_id = note.id
    url = f"/api/v1/notes/{note_id}" if method_path == "id" else method_path

    response = client.delete(url, params=params, headers=_auth(key))

    assert response.status_code in (200, 204), response.text
    assert [p["user_id"] for p in tracing.propagated] == ["acme"]
    [propagated], [update] = tracing.propagated, tracing.span_updates
    attached = {**propagated["metadata"], **update["metadata"]}
    assert attached["scope"] == SCOPE
    assert attached["note_id"] == note_id
    assert attached["passages_deleted"] == 2
    assert PATH not in tracing.everything()
    assert "ada-lovelace" not in tracing.everything()


def test_search_attributes_its_trace_and_never_the_query(client, db_session, tracing):
    key = make_api_key(db_session, "acme")

    response = client.post(
        "/api/v1/search",
        json={"q": "quokkasecret", "top_k": 5, "repo_filter": SCOPE},
        headers=_auth(key),
    )

    assert response.status_code == 200
    assert tracing.propagated == [{"user_id": "acme", "metadata": {"scope": SCOPE}}]
    assert tracing.span_updates == [{"metadata": {"results": 0, "degraded": False}}]
    assert "quokkasecret" not in tracing.everything()


def test_an_unfiltered_search_sends_no_empty_scope(client, db_session, tracing):
    # Langfuse logs a warning for a None metadata value, on every search.
    key = make_api_key(db_session, "acme")

    client.post("/api/v1/search", json={"q": "x", "top_k": 5}, headers=_auth(key))

    assert tracing.propagated == [{"user_id": "acme", "metadata": {}}]


def test_the_real_langfuse_entry_points_accept_what_the_routes_send(db_session, client):
    """No spy: the installed package, with tracing disabled by conftest. Guards the
    API shape, which has changed under this project between Langfuse majors."""
    key = make_api_key(db_session, "acme")

    assert _ingest(client, key).status_code == 200
    assert (
        client.post("/api/v1/search", json={"q": "x", "top_k": 5}, headers=_auth(key)).status_code
        == 200
    )


def test_annotate_span_stays_away_from_langfuse_without_a_recording_span(monkeypatch):
    """With tracing off, update_current_span logs two WARNINGs per call."""

    def forbidden():
        raise AssertionError("get_client was reached with no recording span")

    monkeypatch.setattr(tracing_module, "get_client", forbidden)

    tracing_module.annotate_span(note_id=1)


def test_annotate_span_reaches_a_recording_span(monkeypatch):
    updates = []
    client_stub = SimpleNamespace(update_current_span=lambda **kwargs: updates.append(kwargs))
    monkeypatch.setattr(tracing_module, "get_client", lambda: client_stub)

    with TracerProvider().get_tracer("test").start_as_current_span("span"):
        tracing_module.annotate_span(note_id=1, skipped=False)

    assert updates == [{"metadata": {"note_id": 1, "skipped": False}}]


# --- Settings validators ------------------------------------------------------


def _settings(**overrides):
    return Settings(database_url="postgresql+psycopg://u:p@localhost:5432/db", **overrides)


@pytest.mark.parametrize("value", [-1, 86401])
def test_key_last_used_throttle_out_of_range_fails_at_startup(value):
    with pytest.raises(ValueError, match="KEY_LAST_USED_THROTTLE_S must be in range 0..86400"):
        _settings(key_last_used_throttle_s=value)


@pytest.mark.parametrize("value", [0, 300, 86400])
def test_key_last_used_throttle_in_range_is_accepted(value):
    assert _settings(key_last_used_throttle_s=value).key_last_used_throttle_s == value


@pytest.mark.parametrize("value", [0, 29, 3651])
def test_audit_retention_out_of_range_fails_at_startup(value):
    with pytest.raises(ValueError, match="AUDIT_RETENTION_DAYS must be in range 30..3650"):
        _settings(audit_retention_days=value)


@pytest.mark.parametrize("value", [30, 400, 3650])
def test_audit_retention_in_range_is_accepted(value):
    assert _settings(audit_retention_days=value).audit_retention_days == value


def test_the_audit_defaults_are_the_decided_ones():
    assert Settings.model_fields["key_last_used_throttle_s"].default == 300
    assert Settings.model_fields["audit_retention_days"].default == 400


def test_the_api_exposes_no_route_to_the_audit_table():
    # Operator-only access is a decision, and the absence of a route enforces it.
    paths = TestClient(app).get("/openapi.json").json()["paths"]
    assert not [path for path in paths if "audit" in path or "event" in path]
