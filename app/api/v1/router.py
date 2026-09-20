from fastapi import APIRouter

from app.api.v1 import ingest, search

api_router = APIRouter()
api_router.include_router(search.router, tags=["Search"])
api_router.include_router(ingest.router, tags=["Ingest"])