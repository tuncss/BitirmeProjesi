"""Segmentation task submission endpoint."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

from backend.app.api.dependencies import get_app_settings, get_file_manager
from backend.app.core.exceptions import APIError
from backend.app.tasks.segmentation_task import run_segmentation


router = APIRouter(tags=["segmentation"])


class SegmentRequest(BaseModel):
    """Request body for starting a segmentation job."""

    session_id: str = Field(..., min_length=1)
    model_name: str | None = None


@router.post("/segment", status_code=status.HTTP_202_ACCEPTED, summary="Start segmentation")
async def start_segmentation(request: Request, payload: SegmentRequest) -> dict:
    """Validate an upload session and enqueue an asynchronous segmentation task."""
    settings = get_app_settings(request)
    model_name = payload.model_name or settings.default_model_name
    if model_name not in settings.model_paths:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_MODEL",
            message=f"Invalid model: {model_name}",
            details={"allowed_models": sorted(settings.model_paths)},
        )

    file_manager = get_file_manager(request)
    try:
        modality_map = file_manager.identify_modalities(payload.session_id)
        file_manager.validate_nifti_files(modality_map)
    except ValueError as exc:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_UPLOAD_SESSION",
            message=str(exc),
        ) from exc

    task_id = uuid.uuid4().hex
    try:
        celery_task = run_segmentation.apply_async(
            kwargs={
                "session_id": payload.session_id,
                "model_name": model_name,
                "task_id": task_id,
            },
            task_id=task_id,
        )
        result_dir = file_manager.create_result_dir(task_id)
        _write_task_record(
            result_dir=result_dir,
            task_id=task_id,
            celery_task_id=celery_task.id,
            session_id=payload.session_id,
            model_name=model_name,
        )
    except Exception as exc:  # noqa: BLE001 - expose broker submission failures as API errors
        raise APIError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="TASK_QUEUE_UNAVAILABLE",
            message="Segmentation task could not be submitted.",
            details=str(exc),
        ) from exc

    return {
        "task_id": task_id,
        "celery_task_id": celery_task.id,
        "status": "submitted",
        "model_name": model_name,
        "message": f"Segmentation started with {model_name}.",
    }


def _write_task_record(
    *,
    result_dir: Path,
    task_id: str,
    celery_task_id: str,
    session_id: str,
    model_name: str,
) -> None:
    record = {
        "task_id": task_id,
        "celery_task_id": celery_task_id,
        "session_id": session_id,
        "model_name": model_name,
        "status": "submitted",
    }
    with (result_dir / "task.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
