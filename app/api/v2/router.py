from fastapi import APIRouter, Depends

from app.api.v2 import ingest, notes, search
from app.core.security import get_current_client

api_router = APIRouter(dependencies=[Depends(get_current_client)])
api_router.include_router(search.router, tags=["Search v2"])
api_router.include_router(ingest.router, tags=["Ingest v2"])
api_router.include_router(notes.router, tags=["Notes v2"])
