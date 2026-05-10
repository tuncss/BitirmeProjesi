"""API route aggregation."""

from fastapi import APIRouter

from backend.app.api.routes.download import router as download_router
from backend.app.api.routes.results import router as results_router
from backend.app.api.routes.segment import router as segment_router
from backend.app.api.routes.upload import router as upload_router


api_router = APIRouter()
api_router.include_router(upload_router)
api_router.include_router(segment_router)
api_router.include_router(results_router)
api_router.include_router(download_router)


__all__ = ["api_router"]
