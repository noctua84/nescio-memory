"""Process-wide logging: one line per record, OpenTelemetry field names, stdout.

Delivery is stdout only. In Dash0 a collector's filelog receiver ingests the
container's stdout, so exporting the same records over OTLP as well would
duplicate every one of them. There is deliberately no OTLP log exporter here.

The module is named log_config, not logging, so it cannot shadow the stdlib.
"""
import copy
import json
import logging
import logging.config
import os
import re
import sys
import traceback
from datetime import UTC, datetime
from urllib.parse import unquote

from opentelemetry import trace

from app.config import settings
from app.helper import get_app_version

logger = logging.getLogger(__name__)

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# OpenTelemetry log data model SeverityNumber: the lowest number of each range.
_SEVERITY_NUMBERS = ((50, 21), (40, 17), (30, 13), (20, 9), (10, 5))
_SEVERITY_TRACE = 1

# Defence in depth only. The primary rule is that these values are never passed
# to a logger at all (no request bodies, note content, paths or API keys in
# logs); this catches a future call site that forgets.
_REDACTED_KEYS = frozenset(
    {
        "content",
        "body",
        "q",
        "query",
        "api_key",
        "x_api_key",
        "headers",
        "path_or_uri",
    }
)
_REDACTED = "[redacted]"

# Every attribute a bare LogRecord has, plus the two that Formatter.format adds.
# Anything else on a record was put there by `extra=` and is emitted as a field.
# color_message is uvicorn's ANSI-coloured copy of the message; it is noise here.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "color_message"}

_ACCESS_LOGGER = "uvicorn.access"
# uvicorn passes (client_addr, method, path_with_query_string, http_version,
# status_code) -- see uvicorn/protocols/http/*_impl.py.
_ACCESS_ARG_COUNT = 5
_QUERY_STRING = re.compile(r"\?[^\s\"]*")


def severity_number(levelno: int) -> int:
    """OTel SeverityNumber for a stdlib level; an in-between level maps down."""
    for floor, number in _SEVERITY_NUMBERS:
        if levelno >= floor:
            return number
    return _SEVERITY_TRACE


def _parse_resource_attributes(raw: str, problems: list[str]) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for entry in raw.split(","):
        if not entry.strip():
            continue
        key, separator, value = entry.partition("=")
        key = unquote(key.strip())
        if not separator or not key:
            problems.append(
                "OTEL_RESOURCE_ATTRIBUTES entry is not key=value and was "
                f"skipped: {entry.strip()!r}"
            )
            continue
        attributes[key] = unquote(value.strip())
    return attributes


def _resource_attributes(problems: list[str] | None = None) -> dict[str, str]:
    """The service identity stamped on every record.

    Read from the PROCESS environment, never from .env/Settings: Langfuse's
    OpenTelemetry tracer reads OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES
    from there, and a value only the logger could see would give logs and
    traces two different service identities.

    Precedence is the OTel one: OTEL_SERVICE_NAME, then service.name inside
    OTEL_RESOURCE_ATTRIBUTES, then the app name. Malformed entries are skipped;
    their descriptions are appended to `problems` so the caller can log them
    once the handler they must go through exists.
    """
    collected: list[str] = [] if problems is None else problems
    declared = _parse_resource_attributes(
        os.environ.get("OTEL_RESOURCE_ATTRIBUTES", ""), collected
    )
    name = (
        os.environ.get("OTEL_SERVICE_NAME", "").strip()
        or declared.pop("service.name", "")
        or settings.app_name
    )
    declared.pop("service.name", None)
    version = declared.pop("service.version", "") or get_app_version()
    return {"service.name": name, "service.version": version, **declared}


def _redact_access_record(record: logging.LogRecord) -> logging.LogRecord:
    """A copy of a uvicorn access record with the query string removed.

    uvicorn's message is '<client> - "<METHOD> <path>?<query> HTTP/x" <status>'.
    The query string can carry a note path (DELETE by path_or_uri) or a search
    term, so the raw message is never emitted, in either format. The copy keeps
    other handlers (caplog, a future collector handler) on the original record.
    """
    sanitized = copy.copy(record)
    args = record.args
    if isinstance(args, tuple) and len(args) == _ACCESS_ARG_COUNT:
        client, method, target, _http_version, status = args
        path = str(target).partition("?")[0]
        host, _, port = str(client).rpartition(":")
        sanitized.__dict__["http.request.method"] = method
        sanitized.__dict__["url.path"] = path
        sanitized.__dict__["http.response.status_code"] = status
        sanitized.__dict__["client.address"] = host.strip("[]")
        if port.isdigit():
            sanitized.__dict__["client.port"] = int(port)
        sanitized.msg = f"{method} {path} {status}"
    else:
        # An unexpected shape (a uvicorn upgrade): strip rather than trust it.
        sanitized.msg = _QUERY_STRING.sub("", record.getMessage())
    sanitized.args = None
    return sanitized


class TextFormatter(logging.Formatter):
    """The local-development format, with the same access-log redaction."""

    def format(self, record: logging.LogRecord) -> str:
        if record.name == _ACCESS_LOGGER:
            record = _redact_access_record(record)
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """One JSON object per record, on one line, with OpenTelemetry field names."""

    def __init__(self, resource: dict[str, str] | None = None) -> None:
        super().__init__()
        self._resource = dict(resource or {})

    def format(self, record: logging.LogRecord) -> str:
        if record.name == _ACCESS_LOGGER:
            record = _redact_access_record(record)

        entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "severity_number": severity_number(record.levelno),
            "logger": record.name,
            "message": record.getMessage(),
            **self._resource,
        }

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            entry["trace_id"] = format(span_context.trace_id, "032x")
            entry["span_id"] = format(span_context.span_id, "016x")

        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, _ = record.exc_info
            entry["exception.type"] = exc_type.__qualname__
            entry["exception.message"] = str(exc_value)
            # One string: json.dumps escapes the newlines, so the line stays one.
            entry["exception.stacktrace"] = "".join(
                traceback.format_exception(*record.exc_info)
            )
        elif record.exc_text:
            entry["exception.stacktrace"] = record.exc_text
        if record.stack_info:
            entry["stack_info"] = record.stack_info

        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key in entry:
                # A colliding extra never overwrites a field above.
                continue
            entry[key] = _REDACTED if key in _REDACTED_KEYS else value

        return json.dumps(entry, default=str, ensure_ascii=False)


class _StdoutHandler(logging.StreamHandler):
    """A StreamHandler that looks sys.stdout up on every emit.

    Binding the stream once would pin whatever sys.stdout was at import time,
    which under a test runner's capture is an object that is later closed.
    Same idea as the stdlib's own logging._StderrHandler.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout


def configure_logging(level: str, fmt: str, *, stream=None) -> None:
    """Install the one handler and the logger tree. Safe to call repeatedly.

    Root stays at WARNING so httpx's per-request INFO lines and SQLAlchemy stay
    quiet; `app` and `uvicorn.*` follow `level`. This replaces uvicorn's own
    handlers (uvicorn configures logging in Config.__init__ and imports the app
    later in Config.load(), so this runs second and wins), which is why nothing
    is emitted twice. disable_existing_loggers=False because module loggers such
    as app.core.errors exist before this runs.
    """
    problems: list[str] = []
    formatter: logging.Formatter = (
        JsonFormatter(_resource_attributes(problems))
        if fmt == "json"
        else TextFormatter(TEXT_FORMAT)
    )

    def make_handler() -> logging.Handler:
        return logging.StreamHandler(stream) if stream is not None else _StdoutHandler()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"std": {"()": lambda: formatter}},
            "handlers": {"stdout": {"()": make_handler, "formatter": "std"}},
            "root": {"level": "WARNING", "handlers": ["stdout"]},
            "loggers": {
                "app": {"level": level, "handlers": [], "propagate": True},
                "uvicorn": {
                    "level": level,
                    "handlers": ["stdout"],
                    "propagate": False,
                },
                "uvicorn.error": {"level": level, "handlers": [], "propagate": True},
                "uvicorn.access": {
                    "level": level,
                    "handlers": ["stdout"],
                    "propagate": False,
                },
            },
        }
    )

    for problem in problems:
        logger.warning(problem)
