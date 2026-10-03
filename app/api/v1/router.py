from fastapi import APIRouter, Depends

from app.api.v1 import ingest, search
from app.core.security import get_current_client

api_router = APIRouter(dependencies=[Depends(get_current_client)])
api_router.include_router(search.router, tags=["Search"])
api_router.include_router(ingest.router, tags=["Ingest"])
