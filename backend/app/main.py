"""FastAPI application entry point."""

from __future__ import annotations

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import api_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import APIError, api_error_handler


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI app."""
    app_settings = settings or get_settings()
    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        debug=app_settings.app_debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = app_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(APIError, api_error_handler)

    async def health_check() -> dict:
        cuda_available = torch.cuda.is_available()
        return {
            "status": "healthy",
            "environment": app_settings.app_env,
            "gpu_available": cuda_available,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "device": "cuda" if cuda_available else "cpu",
            "models": {
                name: str(path)
                for name, path in app_settings.model_paths.items()
            },
        }

    app.add_api_route("/health", health_check, methods=["GET"], tags=["health"])
    app.add_api_route(
        f"{app_settings.api_prefix}/health",
        health_check,
        methods=["GET"],
        tags=["health"],
    )

    app.include_router(api_router, prefix=app_settings.api_prefix)

    return app


app = create_app()
