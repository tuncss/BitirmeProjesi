"""Celery application configuration for background segmentation jobs."""

from __future__ import annotations

import sys

from celery import Celery

from backend.app.core.config import get_settings


settings = get_settings()

celery_app = Celery(
    "brain_tumor_segmentation",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["backend.app.tasks.segmentation_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    result_expires=86400,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Celery's default 'prefork' pool relies on os.fork() and is broken on Windows.
# Force the in-process 'solo' pool so `celery -A ... worker` just works locally.
if sys.platform == "win32":
    celery_app.conf.worker_pool = "solo"

