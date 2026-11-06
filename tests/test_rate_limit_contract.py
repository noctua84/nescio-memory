"""The 429 as the OpenAPI document and the README state it.

Also pins the README rows for the /ready 503 and the delete routes' 400s, which
tests/test_openapi_error_responses.py does not list.
"""
from pathlib import Path

import pytest

from app.api.v1.notes import ROOT_SCOPE_CONFLICT_DETAIL, SCOPE_REQUIRED_DETAIL
from app.core.errors import RATE_LIMITED_DETAIL
from app.main import app

README = Path(__file__).resolve().parent.parent / "README.md"
HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


def _operations():
    for path_item in app.openapi()["paths"].values():
        for method, operation in path_item.items():
            if method in HTTP_METHODS:
                yield operation["operationId"], operation


def _error_table_rows() -> list[list[str]]:
    text = README.read_text(encoding="utf-8")
    section = text.split("\n### Error responses\n", 1)[1].split("\n## ", 1)[0]
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("| ") and line[2:5].isdigit()
    ]


def test_429_is_declared_on_ingest_and_search_only():
    declared = {name for name, op in _operations() if "429" in op["responses"]}
    assert declared == {"ingestNote", "searchPassages"}


@pytest.mark.parametrize("operation_id", ["ingestNote", "searchPassages"])
def test_the_429_carries_error_detail_and_retry_after(operation_id):
    response = dict(_operations())[operation_id]["responses"]["429"]

    assert response["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorDetail"
    }
    assert response["headers"]["Retry-After"]["schema"] == {"type": "integer"}


def test_the_readme_error_table_has_the_429_row():
    [row] = [row for row in _error_table_rows() if row[0] == "429"]
    assert row[1] == f"`{RATE_LIMITED_DETAIL}`"
    assert "RATE_LIMIT_INGEST_RETRY_AFTER_S" in row[2]
    assert "window" in row[2]


@pytest.mark.parametrize("detail", [SCOPE_REQUIRED_DETAIL, ROOT_SCOPE_CONFLICT_DETAIL])
def test_the_readme_error_table_has_the_delete_scope_400s(detail):
    assert ["400", f"`{detail}`"] in [row[:2] for row in _error_table_rows()]


def test_the_readme_error_table_has_the_ready_503():
    rows = [row for row in _error_table_rows() if row[0] == "503" and "`GET /ready`" in row[3]]
    assert len(rows) == 1
    assert rows[0][2] == "none"


def test_the_readme_states_what_the_limits_promise():
    text = README.read_text(encoding="utf-8")
    section = text.split("\n### Rate limits\n", 1)[1].split("\n### ", 1)[0]
    for promise in ("hold across", "Skips are free", "fixed", "no off switch", "Retry-After"):
        assert promise in section, promise
    # The hint has to be stated next to what the slot can be held for.
    assert "EMBED_RETRY_BUDGET" in section
