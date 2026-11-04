from fastapi import APIRouter, Depends

from app.api.v1 import ingest, notes, search
from app.core.security import get_current_client
from app.schemas.errors import ErrorDetail

# The error responses every /api/v1 operation shares, declared once on the router
# so the spec cannot drift route by route. A route's own responses still apply
# (notes declare their 404), and FastAPI's 422 for request validation is added
# on its own. The cases behind each code are in app/core/errors.py and
# app/core/security.py; README.md's "Error responses" section lists them.
ERROR_RESPONSES = {
    400: {
        "model": ErrorDetail,
        "description": "The request was rejected. `detail` says why.",
    },
    401: {
        "model": ErrorDetail,
        "description": "The X-API-Key header is missing, unknown or revoked.",
    },
    413: {
        "model": ErrorDetail,
        "description": "The request body is over the size limit.",
    },
    500: {
        "model": ErrorDetail,
        "description": (
            "The embedding backend is misconfigured, or an unexpected error occurred."
        ),
    },
    503: {
        "model": ErrorDetail,
        "description": (
            "A dependency is unavailable, or a query exceeded its time limit. "
            "Retry-After is sent only on the 503s where a retry can succeed; "
            "\"Query exceeded time limit\" sends none."
        ),
        "headers": {
            "Retry-After": {
                "description": "Seconds to wait before retrying.",
                "schema": {"type": "integer"},
            }
        },
    },
}

api_router = APIRouter(
    dependencies=[Depends(get_current_client)],
    responses=ERROR_RESPONSES,
)
api_router.include_router(search.router, tags=["Search"])
api_router.include_router(ingest.router, tags=["Ingest"])
api_router.include_router(notes.router, tags=["Notes"])
