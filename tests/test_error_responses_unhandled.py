"""The catch-all 500 (#70): unhandled exceptions return JSON and leak nothing.

Split from tests/test_error_responses.py by concern. The routes here are
test-only and live on a fresh create_app(), never on the shared `app`.
"""
import json
import logging

import pytest
from fastapi import Depends, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, StatementError

from app.core.db import get_db
from app.core.errors import INTERNAL_ERROR_DETAIL
from app.core.log_config import JsonFormatter
from app.main import create_app

SECRET_TEXT = "secret internal text"
SECRET_PARAM = "SECRET-PARAM-123"


@pytest.fixture
def unhandled_app(db_session):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    return app


@pytest.fixture
def http(unhandled_app):
    # raise_server_exceptions=False: Starlette re-raises after responding, and
    # the default client would raise it here instead of returning the response.
    return TestClient(unhandled_app, raise_server_exceptions=False)


@pytest.fixture
def error_log(caplog):
    """Formatted JSON lines of ERROR records from app.core.errors."""
    formatter = JsonFormatter()
    with caplog.at_level(logging.ERROR, logger="app.core.errors"):
        yield lambda: [
            formatter.format(r)
            for r in caplog.records
            if r.name == "app.core.errors" and r.levelno >= logging.ERROR
        ]


def test_unhandled_exception_returns_json_500(unhandled_app, http, error_log):
    @unhandled_app.get("/boom")
    def boom():
        raise RuntimeError(SECRET_TEXT)

    response = http.get("/boom")

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert response.text == '{"detail":"Internal server error"}'
    assert SECRET_TEXT not in response.text
    assert "retry-after" not in response.headers

    lines = error_log()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["message"] == "Unhandled RuntimeError on GET /boom"
    assert entry["exception.type"] == "RuntimeError"
    # A generic exception keeps its text in the log, never in the response.
    assert entry["exception.message"] == SECRET_TEXT
    assert "Traceback" in entry["exception.stacktrace"]
    assert "in boom" in entry["exception.stacktrace"]


def test_exception_in_a_dependency_is_covered(unhandled_app, http, error_log):
    def broken_dependency():
        raise RuntimeError(SECRET_TEXT)

    @unhandled_app.get("/dep", dependencies=[Depends(broken_dependency)])
    def dep():
        return {"ok": True}

    response = http.get("/dep")

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert len(error_log()) == 1


def test_existing_mappings_are_unchanged(unhandled_app, http, error_log):
    @unhandled_app.get("/teapot")
    def teapot():
        raise HTTPException(status_code=418, detail="short and stout")

    @unhandled_app.get("/typed/{number}")
    def typed(number: int):
        return {"number": number}

    assert http.get("/teapot").status_code == 418
    assert http.get("/teapot").json() == {"detail": "short and stout"}
    assert http.get("/does-not-exist").status_code == 404
    assert http.get("/does-not-exist").json() == {"detail": "Not Found"}
    validation = http.get("/typed/not-a-number")
    assert validation.status_code == 422
    assert isinstance(validation.json()["detail"], list)
    assert error_log() == []


def test_hand_built_statement_error_hides_parameters(unhandled_app, http, error_log):
    @unhandled_app.get("/stmt")
    def stmt():
        # hide_parameters left at its default: the handler must not rely on the
        # engine having been configured.
        raise StatementError(
            "boom",
            "INSERT INTO notes (content) VALUES (%(content)s)",
            {"content": SECRET_PARAM},
            ValueError(f"bad value {SECRET_PARAM}"),
        )

    response = http.get("/stmt")

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert SECRET_PARAM not in response.text
    lines = error_log()
    assert len(lines) == 1
    assert SECRET_PARAM not in lines[0]
    assert "INSERT INTO notes" in json.loads(lines[0])["exception.stacktrace"]


def test_driver_message_echoing_a_value_does_not_reach_the_log(
    unhandled_app, http, error_log, db_session
):
    """Postgres quotes the offending key in a unique violation's DETAIL.

    hide_parameters (which the test engine here does not even set) would not
    remove that: it is the driver's message, not the bound parameters.
    """

    @unhandled_app.get("/dup")
    def dup():
        db_session.execute(text("CREATE TEMP TABLE echo_probe (v text PRIMARY KEY)"))
        statement = text("INSERT INTO echo_probe (v) VALUES (:v)")
        db_session.execute(statement, {"v": SECRET_PARAM})
        db_session.execute(statement, {"v": SECRET_PARAM})

    # Proves the premise: without the scrub the raw error does carry the value.
    with pytest.raises(DBAPIError) as raw:
        dup()
    assert SECRET_PARAM in str(raw.value)
    db_session.rollback()

    response = http.get("/dup")

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
    assert SECRET_PARAM not in response.text
    lines = error_log()
    assert len(lines) == 1
    assert SECRET_PARAM not in lines[0]
    entry = json.loads(lines[0])
    assert "SQLSTATE 23505" in entry["exception.message"]
    assert "INSERT INTO echo_probe" in entry["exception.stacktrace"]


def test_data_error_handler_scrubs_the_logged_driver_message(
    unhandled_app, http, caplog, db_session
):
    """The 400 path logs exc_info too; Postgres echoes the value in 22P02."""

    @unhandled_app.get("/bad-int")
    def bad_int():
        db_session.execute(text("SELECT CAST(:p AS integer)"), {"p": SECRET_PARAM})

    formatter = JsonFormatter()
    with caplog.at_level(logging.WARNING, logger="app.core.errors"):
        response = http.get("/bad-int")

    assert response.status_code == 400
    assert SECRET_PARAM not in response.text
    lines = [
        formatter.format(r) for r in caplog.records if r.name == "app.core.errors"
    ]
    assert lines
    assert all(SECRET_PARAM not in line for line in lines)
