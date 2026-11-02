"""Structured logging (#29): one JSON line per record, OTel field names, stdout.

The formatter tests build their own logger and handler, so they say nothing
about global state. The configure_logging tests swap the process-wide logging
tree for one that writes into a StringIO and restore the real one afterwards.
"""
import io
import json
import logging
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from alembic.config import Config
from opentelemetry.sdk.trace import TracerProvider
from pydantic import ValidationError

from alembic import command
from app.config import Settings, settings
from app.core import errors as errors_module
from app.core.log_config import (
    JsonFormatter,
    TextFormatter,
    _resource_attributes,
    configure_logging,
    severity_number,
)
from app.main import app
from tests.conftest import ALEMBIC_INI

RESOURCE = {"service.name": "svc", "service.version": "1.2.3"}
FIELDS = {
    "timestamp",
    "level",
    "severity_number",
    "logger",
    "message",
    "service.name",
    "service.version",
}


def _isolated_logger(formatter: logging.Formatter) -> tuple[logging.Logger, io.StringIO]:
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(formatter)
    logger = logging.getLogger("nescio_test.formatter")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, buffer


def _json_lines(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


@pytest.fixture
def json_logger():
    return _isolated_logger(JsonFormatter(RESOURCE))


@pytest.fixture
def configured():
    """The real logging tree, pointed at a StringIO; restored afterwards."""

    def configure(level="INFO", fmt="json") -> io.StringIO:
        buffer = io.StringIO()
        configure_logging(level, fmt, stream=buffer)
        return buffer

    yield configure
    configure_logging(settings.log_level, settings.log_format)


# --- formatter ---------------------------------------------------------------


def test_a_record_is_one_json_line_with_every_field(json_logger):
    logger, buffer = json_logger
    logger.warning("hello %s", "world", extra={"note_id": 7})

    assert buffer.getvalue().count("\n") == 1
    (entry,) = _json_lines(buffer)
    assert FIELDS <= entry.keys()
    assert entry["level"] == "WARNING"
    assert entry["severity_number"] == 13
    assert entry["logger"] == "nescio_test.formatter"
    assert entry["message"] == "hello world"
    assert entry["service.name"] == "svc"
    assert entry["service.version"] == "1.2.3"
    assert entry["note_id"] == 7
    # ISO 8601, UTC, millisecond precision.
    assert entry["timestamp"].endswith("Z") and len(entry["timestamp"]) == 24


@pytest.mark.parametrize(
    ("levelno", "expected"),
    [
        (logging.DEBUG, 5),
        (logging.INFO, 9),
        (logging.WARNING, 13),
        (logging.ERROR, 17),
        (logging.CRITICAL, 21),
        (25, 9),  # between INFO and WARNING: nearest lower
        (5, 1),  # below DEBUG (uvicorn's TRACE): OTel TRACE
    ],
)
def test_severity_number_follows_the_otel_mapping(levelno, expected):
    assert severity_number(levelno) == expected


def test_an_exception_is_three_fields_on_one_line(json_logger):
    logger, buffer = json_logger
    try:
        raise ValueError("boom\nsecond line")
    except ValueError:
        logger.exception("failed")

    assert buffer.getvalue().count("\n") == 1
    (entry,) = _json_lines(buffer)
    assert entry["exception.type"] == "ValueError"
    assert entry["exception.message"] == "boom\nsecond line"
    assert entry["exception.stacktrace"].startswith("Traceback")
    assert "ValueError: boom" in entry["exception.stacktrace"]
    assert entry["level"] == "ERROR"


def test_stack_info_stays_on_one_line(json_logger):
    logger, buffer = json_logger
    logger.warning("where", stack_info=True)

    assert buffer.getvalue().count("\n") == 1
    assert "Stack (most recent call last)" in _json_lines(buffer)[0]["stack_info"]


def test_trace_ids_appear_inside_an_active_span_and_not_outside(json_logger):
    logger, buffer = json_logger
    tracer = TracerProvider().get_tracer("test")

    logger.info("before")
    with tracer.start_as_current_span("work") as span:
        logger.info("inside")
        context = span.get_span_context()
    logger.info("after")

    before, inside, after = _json_lines(buffer)
    assert inside["trace_id"] == format(context.trace_id, "032x")
    assert inside["span_id"] == format(context.span_id, "016x")
    assert len(inside["trace_id"]) == 32 and len(inside["span_id"]) == 16
    for outside in (before, after):
        assert "trace_id" not in outside and "span_id" not in outside


def test_denylisted_extras_are_redacted(json_logger):
    logger, buffer = json_logger
    logger.info(
        "x",
        extra={
            "content": "note body",
            "query": "search term",
            "api_key": "nm_secret",
            "path_or_uri": "private/notes.md",
            "note_id": 3,
        },
    )

    raw = buffer.getvalue()
    for leaked in ("note body", "search term", "nm_secret", "private/notes.md"):
        assert leaked not in raw
    entry = _json_lines(buffer)[0]
    assert entry["content"] == entry["query"] == "[redacted]"
    assert entry["note_id"] == 3


def test_an_extra_cannot_overwrite_a_core_field(json_logger):
    logger, buffer = json_logger
    logger.info("real", extra={"level": "FAKE", "severity_number": 99})

    entry = _json_lines(buffer)[0]
    assert entry["level"] == "INFO" and entry["severity_number"] == 9


def test_an_unserialisable_extra_does_not_break_the_line(json_logger):
    logger, buffer = json_logger
    logger.info("x", extra={"thing": object()})

    assert "object object" in _json_lines(buffer)[0]["thing"]


# --- service identity --------------------------------------------------------


@pytest.fixture
def clean_otel_env(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    return monkeypatch


def test_service_name_defaults_to_the_app_name(clean_otel_env):
    resource = _resource_attributes()

    assert resource["service.name"] == settings.app_name
    assert resource["service.version"]


def test_service_name_precedence_is_otel_then_resource_attributes_then_app_name(
    clean_otel_env,
):
    clean_otel_env.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=from-attrs")
    assert _resource_attributes()["service.name"] == "from-attrs"

    clean_otel_env.setenv("OTEL_SERVICE_NAME", "from-service-name")
    resource = _resource_attributes()
    assert resource["service.name"] == "from-service-name"
    assert list(resource).count("service.name") == 1


def test_resource_attributes_are_parsed_and_percent_decoded(clean_otel_env):
    clean_otel_env.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "deployment.environment=prod,team=a%20b,service.version=9.9.9",
    )

    resource = _resource_attributes()

    assert resource["deployment.environment"] == "prod"
    assert resource["team"] == "a b"
    assert resource["service.version"] == "9.9.9"


def test_a_malformed_resource_attribute_is_skipped_with_one_warning(
    clean_otel_env, configured
):
    clean_otel_env.setenv("OTEL_RESOURCE_ATTRIBUTES", "good=1,nonsense,=novalue,ok=2")

    buffer = configured()

    warnings = [
        e for e in _json_lines(buffer) if e["logger"] == "app.core.log_config"
    ]
    assert len(warnings) == 2
    assert all(e["level"] == "WARNING" for e in warnings)
    entry = _resource_attributes([])
    assert entry["good"] == "1" and entry["ok"] == "2"
    assert "nonsense" not in entry


# --- configure_logging -------------------------------------------------------


def test_a_warning_from_app_core_errors_reaches_the_handler_as_one_json_line(
    clean_otel_env, configured
):
    buffer = configured()

    errors_module.logger.warning("database unavailable", extra={"attempt": 2})

    assert buffer.getvalue().count("\n") == 1
    (entry,) = _json_lines(buffer)
    assert FIELDS <= entry.keys()
    assert entry["logger"] == "app.core.errors"
    assert entry["message"] == "database unavailable"
    assert entry["attempt"] == 2


def test_log_level_debug_lets_app_debug_through_but_not_httpx_info(configured):
    buffer = configured("DEBUG")
    logging.getLogger("app.anything").debug("app debug")
    logging.getLogger("httpx").info("httpx info")
    assert [e["message"] for e in _json_lines(buffer)] == ["app debug"]

    buffer = configured("INFO")
    logging.getLogger("app.anything").debug("app debug")
    logging.getLogger("app.anything").info("app info")
    assert [e["message"] for e in _json_lines(buffer)] == ["app info"]


def test_configuring_twice_does_not_duplicate_records(configured):
    configured()
    buffer = configured()

    logging.getLogger("app.anything").warning("once")

    assert len(buffer.getvalue().splitlines()) == 1


def test_log_format_text_switches_the_format(configured):
    buffer = configured(fmt="text")

    logging.getLogger("app.anything").warning("plain words")

    (line,) = buffer.getvalue().splitlines()
    assert line.endswith("WARNING app.anything: plain words")
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_the_access_line_has_no_query_string(configured, fmt):
    buffer = configured(fmt=fmt)

    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:50321",
        "DELETE",
        "/api/v1/notes?path_or_uri=private/secret.md",
        "1.1",
        204,
    )

    assert "secret" not in buffer.getvalue()
    (line,) = buffer.getvalue().splitlines()
    if fmt == "json":
        entry = json.loads(line)
        assert entry["message"] == "DELETE /api/v1/notes 204"
        assert entry["http.request.method"] == "DELETE"
        assert entry["url.path"] == "/api/v1/notes"
        assert entry["http.response.status_code"] == 204
        assert entry["client.address"] == "127.0.0.1"
        assert entry["client.port"] == 50321
    else:
        assert line.endswith("uvicorn.access: DELETE /api/v1/notes 204")


def test_an_access_record_of_an_unexpected_shape_is_still_stripped(configured):
    buffer = configured()

    logging.getLogger("uvicorn.access").info('GET /x?token=secret HTTP/1.1 200')

    assert "secret" not in buffer.getvalue()


def test_the_redaction_does_not_alter_the_record_other_handlers_see():
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, "%s %s %s %s %d",
        ("1.2.3.4:5", "GET", "/p?q=secret", "1.1", 200), None,
    )

    TextFormatter().format(record)

    assert record.args[2] == "/p?q=secret"


def test_a_real_uvicorn_request_gives_one_access_line_without_the_query(configured):
    """uvicorn configures logging itself first; ours must replace that."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off")
    buffer = configured()  # after uvicorn.Config(): the app is imported later
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        response = httpx.get(f"http://127.0.0.1:{port}/health?path_or_uri=secret")
        assert response.status_code == 200
        while "uvicorn.access" not in buffer.getvalue() and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    entries = _json_lines(buffer)
    access = [e for e in entries if e["logger"] == "uvicorn.access"]
    assert len(access) == 1
    assert access[0]["url.path"] == "/health"
    assert access[0]["http.response.status_code"] == 200
    assert "secret" not in buffer.getvalue()
    assert "color_message" not in buffer.getvalue()
    # Nothing was emitted twice, whichever uvicorn logger it came from.
    messages = [(e["logger"], e["message"], e["timestamp"]) for e in entries]
    assert len(messages) == len(set(messages))


# --- settings ----------------------------------------------------------------


def _settings(**overrides) -> Settings:
    return Settings(database_url="postgresql://u:p@localhost/db", _env_file=None, **overrides)


def test_log_level_is_normalised_to_upper_case():
    assert _settings(log_level="debug").log_level == "DEBUG"


@pytest.mark.parametrize("bad", ["verbose", "WARN", "NOTSET", ""])
def test_an_invalid_log_level_fails_at_construction_naming_the_variable(bad):
    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        _settings(log_level=bad)


def test_log_format_defaults_to_json_and_rejects_other_values():
    assert _settings().log_format == "json"
    assert _settings(log_format="text").log_format == "text"
    with pytest.raises(ValidationError, match="log_format"):
        _settings(log_format="xml")


# --- alembic -----------------------------------------------------------------


def test_running_the_alembic_env_does_not_disable_app_loggers(configured):
    # `configured` restores the logging tree: fileConfig replaces root's handlers.
    app_logger = logging.getLogger("app.core.errors")
    assert app_logger is errors_module.logger
    assert not app_logger.disabled

    # Offline mode runs env.py (and so fileConfig) without needing a database.
    command.upgrade(Config(str(ALEMBIC_INI)), "head", sql=True)

    assert not app_logger.disabled
