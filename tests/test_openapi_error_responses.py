"""The error contract, as the OpenAPI document and the README both state it.

Two halves, one contract:

- every /api/v1 operation in app.openapi() declares the shared error responses
  that app/api/v1/router.py sets on the router, 503 carries Retry-After, and the
  per-route 404s survive the merge;
- the README "Error responses" table agrees with the constants and classes in
  app/core/errors.py, so a changed detail string or retry rule fails here until
  the table is updated.
"""
import re
from pathlib import Path

import pytest

from app.core.errors import (
    DATABASE_UNAVAILABLE_DETAIL,
    INTERNAL_ERROR_DETAIL,
    INVALID_DATA_DETAIL,
    QUERY_TIMEOUT_DETAIL,
    RETRY_AFTER_SECONDS,
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.main import app

README = Path(__file__).resolve().parent.parent / "README.md"

HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
SHARED_ERROR_CODES = ("400", "401", "413", "500", "503")
ERROR_SCHEMA_REF = {"$ref": "#/components/schemas/ErrorDetail"}


def _api_v1_operations():
    """(name, operation) for every operation under /api/v1 in the spec."""
    spec = app.openapi()
    for path, path_item in spec["paths"].items():
        if not path.startswith("/api/v1/"):
            continue
        for method, operation in path_item.items():
            if method in HTTP_METHODS:
                yield f"{method.upper()} {path}", operation


def _operations_by_id():
    return {operation["operationId"]: operation for _, operation in _api_v1_operations()}


def _json_schema(response):
    return response.get("content", {}).get("application/json", {}).get("schema")


def test_there_are_api_v1_operations_to_check():
    # Guards the loops below against passing vacuously if the prefix ever changes.
    assert list(_api_v1_operations()), "no /api/v1 operations found in app.openapi()"


@pytest.mark.parametrize("code", SHARED_ERROR_CODES)
def test_every_api_v1_operation_declares_the_shared_error_response(code):
    missing = [
        name
        for name, operation in _api_v1_operations()
        if _json_schema(operation["responses"].get(code, {})) != ERROR_SCHEMA_REF
    ]
    assert not missing, f"{code} not declared with ErrorDetail on: {missing}"


def test_every_api_v1_operation_declares_retry_after_on_503():
    missing = [
        name
        for name, operation in _api_v1_operations()
        if "Retry-After" not in operation["responses"]["503"].get("headers", {})
    ]
    assert not missing, f"503 without a Retry-After header on: {missing}"


def test_error_detail_schema_is_a_required_string_detail():
    schema = app.openapi()["components"]["schemas"]["ErrorDetail"]
    assert schema["required"] == ["detail"]
    assert schema["properties"]["detail"]["type"] == "string"


@pytest.mark.parametrize("operation_id", ["getNote", "deleteNote"])
def test_the_per_route_404_survives_the_router_level_declarations(operation_id):
    # The router declares no 404, so the note routes' own 404 must still be
    # there after the merge, and the shared codes must be there beside it.
    responses = _operations_by_id()[operation_id]["responses"]
    assert responses["404"]["description"] == "No such note for this client."
    assert _json_schema(responses["401"]) == ERROR_SCHEMA_REF


# ---- README ----------------------------------------------------------------

def _error_section() -> str:
    text = README.read_text(encoding="utf-8")
    assert "\n### Error responses\n" in text, "README has no 'Error responses' section"
    return text.split("\n### Error responses\n", 1)[1].split("\n## ", 1)[0]


def _error_rows() -> list[tuple[int, str, str]]:
    """(status, detail, retry cell) for each table row whose detail is a code span."""
    rows = []
    for line in _error_section().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.fullmatch(r"\d{3}", cells[0]) and re.fullmatch(r"`[^`]+`", cells[1]):
            rows.append((int(cells[0]), cells[1][1:-1], cells[2]))
    return rows


def _row(detail: str) -> tuple[int, str, str]:
    matches = [row for row in _error_rows() if row[1] == detail]
    assert len(matches) == 1, f"expected exactly one README row for {detail!r}, found {len(matches)}"
    return matches[0]


# Each README row for a constant in app/core/errors.py. The status and retry
# rule come from the class attributes where a class exists, so the table cannot
# quietly disagree with the mapping.
ERRORS_PY_CASES = [
    pytest.param(DATABASE_UNAVAILABLE_DETAIL, 503, True, id="database-unavailable"),
    pytest.param(QUERY_TIMEOUT_DETAIL, 503, False, id="query-timeout"),
    pytest.param(INVALID_DATA_DETAIL, 400, False, id="invalid-data"),
    pytest.param(INTERNAL_ERROR_DETAIL, 500, False, id="internal-error"),
    pytest.param(
        EmbeddingBackendError.detail,
        EmbeddingBackendError.status_code,
        EmbeddingBackendError.retry_after,
        id="embedding-unavailable",
    ),
    pytest.param(
        EmbeddingBackendBadResponse.detail,
        EmbeddingBackendBadResponse.status_code,
        EmbeddingBackendBadResponse.retry_after,
        id="embedding-bad-response",
    ),
    pytest.param(
        EmbeddingBackendMisconfigured.detail,
        EmbeddingBackendMisconfigured.status_code,
        EmbeddingBackendMisconfigured.retry_after,
        id="embedding-misconfigured",
    ),
]


@pytest.mark.parametrize("detail, status_code, retry_after", ERRORS_PY_CASES)
def test_readme_row_matches_the_error_mapping(detail, status_code, retry_after):
    status, _, retry_cell = _row(detail)
    assert status == status_code
    assert retry_cell == (str(RETRY_AFTER_SECONDS) if retry_after else "none")


def test_readme_documents_the_retry_after_rule_once():
    # The rule is stated once, in the prose above the table; a second statement
    # would be a second place to fall out of step with the code.
    section = _error_section()
    assert section.count("Retry-After") >= 1
    assert section.count("sent only when a retry can succeed") == 1
