"""Shared FastAPI dependencies for API routes."""

from __future__ import annotations

from fastapi import Request

from backend.app.core.config import Settings, get_settings
from backend.app.services.file_manager import FileManager


def get_app_settings(request: Request) -> Settings:
    """Return app-local settings when create_app injected them."""
    return getattr(request.app.state, "settings", None) or get_settings()


def get_file_manager(request: Request) -> FileManager:
    """Create a file manager bound to the current app settings."""
    return FileManager(get_app_settings(request))

