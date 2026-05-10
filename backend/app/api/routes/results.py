"""Result polling endpoint for segmentation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from celery.result import AsyncResult
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from backend.app.api.dependencies import get_app_settings
from backend.app.core.exceptions import TaskNotFoundError
from backend.app.tasks.celery_app import celery_app


router = APIRouter(tags=["results"])
RESULT_FILE_WHITELIST = {
    "segmentation.nii.gz",
    "ground_truth.nii.gz",
    "background.nii.gz",
}


@router.get("/results/{task_id}", summary="Get segmentation task status and results")
async def get_results(request: Request, task_id: str) -> dict[str, Any]:
    """Return task progress, failure details, or persisted segmentation metadata."""
    settings = get_app_settings(request)
    task_dir = Path(settings.results_dir) / task_id
    metadata_path = task_dir / "metadata.json"
    failure_path = task_dir / "failure.json"
    task_record_path = task_dir / "task.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "results": metadata,
        }

    failure_record = _read_task_record(failure_path)
    if failure_record is not None:
        return {
            "task_id": task_id,
            "status": "failed",
            "progress": failure_record.get("progress", 100),
            "step": failure_record.get("step", "error"),
            "message": failure_record.get("message", "Segmentation failed."),
            "error": failure_record.get("error", failure_record.get("message")),
        }

    celery_result = AsyncResult(task_id, app=celery_app)
    state = celery_result.state
    info = celery_result.info if isinstance(celery_result.info, dict) else {}
    task_record = _read_task_record(task_record_path)

    if state == "PENDING" and task_record is None:
        raise TaskNotFoundError(task_id)

    if state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "failed",
            "progress": info.get("progress", 100),
            "step": info.get("step", "error"),
            "message": info.get("message", "Segmentation failed."),
            "error": info.get("error") or str(celery_result.info),
        }

    if state in {"PROCESSING", "STARTED"}:
        return {
            "task_id": task_id,
            "status": "processing",
            "progress": info.get("progress", 0),
            "step": info.get("step", "processing"),
            "message": info.get("message", "Segmentation is running."),
        }

    response = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "step": "queued",
        "message": "Segmentation task is queued or waiting to start.",
    }
    if task_record is not None:
        response["model_name"] = task_record.get("model_name")
        response["celery_task_id"] = task_record.get("celery_task_id")
    return response


@router.get("/results/{task_id}/files/{filename:path}", summary="Download a whitelisted result file")
async def get_result_file(request: Request, task_id: str, filename: str) -> FileResponse:
    """Stream a whitelisted saved result artifact."""
    safe_filename = Path(filename).name
    if filename != safe_filename or safe_filename not in RESULT_FILE_WHITELIST:
        raise TaskNotFoundError(task_id)

    settings = get_app_settings(request)
    file_path = Path(settings.results_dir) / task_id / safe_filename
    if not file_path.is_file():
        raise TaskNotFoundError(task_id)

    return FileResponse(
        path=str(file_path),
        filename=safe_filename,
        media_type="application/octet-stream",
    )


def _read_task_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
