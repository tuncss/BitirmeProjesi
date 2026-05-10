"""Upload endpoints for BraTS NIfTI files."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile, status

from backend.app.api.dependencies import get_app_settings, get_file_manager
from backend.app.core.exceptions import APIError


router = APIRouter(tags=["upload"])


@router.post("/upload", status_code=status.HTTP_201_CREATED, summary="Upload NIfTI modality files")
async def upload_files(request: Request, files: list[UploadFile] = File(...)) -> dict:
    """Upload four BraTS modalities as NIfTI files or a zip containing them."""
    if not files:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="NO_FILES_UPLOADED",
            message="At least one NIfTI or ZIP file must be uploaded.",
        )

    settings = get_app_settings(request)
    file_manager = get_file_manager(request)
    session_id = file_manager.create_upload_session()
    saved_files: list[Path] = []
    extracted_files: list[Path] = []

    for upload in files:
        filename = upload.filename or ""
        content = await upload.read()
        if len(content) > settings.max_upload_size_bytes:
            raise APIError(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                code="FILE_TOO_LARGE",
                message=f"File too large: {filename}",
                details={"max_upload_size_mb": settings.max_upload_size_mb},
            )

        try:
            saved = await file_manager.save_uploaded_file(session_id, filename, content)
            saved_files.append(saved)
            if saved.name.lower().endswith(".zip"):
                extracted_files.extend(file_manager.extract_zip(session_id, saved))
        except ValueError as exc:
            raise APIError(
                status_code=status.HTTP_400_BAD_REQUEST,
                code="INVALID_UPLOAD",
                message=str(exc),
            ) from exc

    try:
        modality_map = file_manager.identify_modalities(session_id)
        file_manager.validate_nifti_files(modality_map)
    except ValueError as exc:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_UPLOAD",
            message=str(exc),
        ) from exc

    return {
        "session_id": session_id,
        "status": "uploaded",
        "modalities": {modality: str(path) for modality, path in modality_map.items()},
        "uploaded_files": [path.name for path in saved_files],
        "extracted_files": [path.name for path in extracted_files],
        "message": "Files uploaded successfully.",
    }
