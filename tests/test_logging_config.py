"""Unit tests for app/core/logging_config.py.

configure_logging() mutates process-global logger state (logging.getLogger
returns the same object every time), so every test here restores whatever it
touched. Most of these tests exercise configure_logging() directly and don't
need a database; the one exception is
test_alembic_fileconfig_does_not_disable_the_app_logger_hierarchy, which pulls
in the session-scoped `engine` fixture specifically to run migrations
in-process the way the real test suite does.
"""
import io
import logging
import re

import pytest

from app.core.logging_config import _HANDLER_NAME, configure_logging


@pytest.fixture(autouse=True)
def _restore_app_logger_state():
    """Snapshot and restore the "app" logger around every test in this file.

    configure_logging() is only ever meant to run once, at create_app() time,
    in a real process. Calling it from a test to observe its effects mutates
    global state that other tests (and the `client` fixture's app, already
    configured at import time) depend on, so each test must leave the logger
    exactly as it found it.
    """
    logger = logging.getLogger("app")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate

    yield

    logger.setLevel(original_level)
    logger.handlers = original_handlers
    logger.propagate = original_propagate


def _our_handler(logger: logging.Logger) -> logging.Handler:
    for handler in logger.handlers:
        if handler.get_name() == _HANDLER_NAME:
            return handler
    raise AssertionError(f"no handler named {_HANDLER_NAME!r} on {logger.name!r}")


def test_at_info_level_a_debug_record_is_not_emitted_but_an_info_record_is():
    configure_logging("INFO")
    handler = _our_handler(logging.getLogger("app"))
    stream = io.StringIO()
    handler.setStream(stream)

    child = logging.getLogger("app.something")
    child.debug("should not appear")
    child.info("should appear")

    output = stream.getvalue()
    assert "should not appear" not in output
    assert "should appear" in output


def test_at_debug_level_a_debug_record_is_emitted():
    configure_logging("DEBUG")
    handler = _our_handler(logging.getLogger("app"))
    stream = io.StringIO()
    handler.setStream(stream)

    logging.getLogger("app.something").debug("now visible")

    assert "now visible" in stream.getvalue()


def test_a_warning_reaches_the_handler_stream_formatted_with_a_timestamp():
    configure_logging("INFO")
    handler = _our_handler(logging.getLogger("app"))
    stream = io.StringIO()
    handler.setStream(stream)

    logging.getLogger("app.core.errors").warning("statement timeout on /api/v1/search")

    line = stream.getvalue().strip()
    # "%(asctime)s %(levelname)s %(name)s: %(message)s" -- the approved format
    # from the discovery brief. The timestamp format itself (down to
    # milliseconds) is asserted, not just that *a* line was logged.
    assert re.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} WARNING app\.core\.errors: "
        r"statement timeout on /api/v1/search$",
        line,
    )


def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    configure_logging("DEBUG")

    logger = logging.getLogger("app")
    ours = [h for h in logger.handlers if h.get_name() == _HANDLER_NAME]
    assert len(ours) == 1
    assert logger.level == logging.DEBUG


def test_a_foreign_handler_on_the_app_logger_survives_reconfiguration():
    logger = logging.getLogger("app")
    foreign = logging.NullHandler()
    foreign.set_name("someone-elses-handler")
    logger.addHandler(foreign)

    configure_logging("INFO")

    assert foreign in logger.handlers


def test_configure_logging_does_not_touch_the_root_logger():
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    original_propagate = root.propagate

    configure_logging("DEBUG")

    assert root.level == original_level
    assert root.handlers == original_handlers
    assert root.propagate == original_propagate


@pytest.mark.parametrize("uvicorn_logger_name", ["uvicorn", "uvicorn.error"])
def test_configure_logging_does_not_touch_uvicorn_loggers(uvicorn_logger_name):
    uvicorn_logger = logging.getLogger(uvicorn_logger_name)
    original_level = uvicorn_logger.level
    original_handlers = list(uvicorn_logger.handlers)
    original_propagate = uvicorn_logger.propagate

    configure_logging("DEBUG")

    assert uvicorn_logger.level == original_level
    assert uvicorn_logger.handlers == original_handlers
    assert uvicorn_logger.propagate == original_propagate


def test_the_app_logger_does_not_propagate_to_the_root_logger():
    configure_logging("INFO")
    assert logging.getLogger("app").propagate is False


def test_alembic_fileconfig_does_not_disable_the_app_logger_hierarchy(engine):
    # Regression test for alembic/env.py's disable_existing_loggers=False
    # (this used to be worked around in test_error_responses.py by
    # re-enabling the logger by hand): logging.config.fileConfig() defaults to disable_existing_loggers=True,
    # which disables every logger that already exists and isn't named in
    # alembic.ini -- including "app" and "app.core.errors", created at import
    # time when conftest imports app.main. The session-scoped `engine`
    # fixture runs `alembic upgrade head` in-process, so requesting it here
    # is what actually exercises alembic/env.py's fileConfig call.
    assert logging.getLogger("app").disabled is False
    assert logging.getLogger("app.core.errors").disabled is False
