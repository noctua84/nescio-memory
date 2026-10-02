"""Configure the app's own logger hierarchy.

Only ``logging.getLogger("app")`` and its children are touched here -- never
the root logger, and never ``uvicorn.*``, ``sqlalchemy``, ``httpx``, or
``langfuse``. Uvicorn owns and configures its own loggers (its
``--log-level``/``--log-config`` govern those); this module has no business
changing them, and LOG_LEVEL is documented as governing only the app's own
logs.
"""
import logging
import sys

# Tag our handler by name so configure_logging can find and remove exactly
# the handler it previously installed, without touching any handler a
# different piece of code (or a test) attached to the "app" logger.
_HANDLER_NAME = "app.core.logging_config.handler"

_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


def configure_logging(level: str) -> None:
    """Set the level and handler for the "app" logger hierarchy.

    Idempotent: calling this more than once (e.g. because create_app() can
    run more than once in tests) replaces the previously installed handler
    rather than stacking duplicates.
    """
    logger = logging.getLogger("app")
    logger.setLevel(level)

    for handler in list(logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(_FORMATTER)
    logger.addHandler(handler)

    # The root logger may end up with its own handler -- alembic's fileConfig
    # when migrations run in-process (as the test suite does), or a process
    # manager wiring one up -- and propagating would then print every app
    # log line twice. Consequence: pytest's caplog fixture attaches its
    # handler to the root logger by default, so tests that want to assert on
    # app.* log records must attach caplog to the "app" logger explicitly
    # (e.g. `caplog.set_level(..., logger="app")` or
    # `logging.getLogger("app").addHandler(caplog.handler)`).
    logger.propagate = False
