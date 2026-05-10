"""Background task queue package."""

from backend.app.tasks.celery_app import celery_app


__all__ = ["celery_app"]
