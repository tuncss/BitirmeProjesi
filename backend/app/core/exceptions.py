"""Shared API exception types and handlers."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class APIError(HTTPException):
    """Base HTTP exception that carries a stable machine-readable error code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(
            status_code=status_code,
            detail={
                "code": code,
                "message": message,
                "details": details,
            },
        )


class UploadedFileNotFoundError(APIError):
    def __init__(self, file_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="FILE_NOT_FOUND",
            message=f"File not found: {file_id}",
        )


class InvalidFileFormatError(APIError):
    def __init__(self, filename: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_FILE_FORMAT",
            message=f"Invalid file format: {filename}. Supported: .nii, .nii.gz, .zip",
        )


class ModelNotFoundError(APIError):
    def __init__(self, model_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="MODEL_NOT_FOUND",
            message=f"Model not found: {model_name}",
        )


class TaskNotFoundError(APIError):
    def __init__(self, task_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="TASK_NOT_FOUND",
            message=f"Task not found: {task_id}",
        )


class SegmentationError(APIError):
    def __init__(self, message: str, details: Any | None = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="SEGMENTATION_ERROR",
            message=message,
            details=details,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Render project API errors in a consistent envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
            "meta": {
                "path": request.url.path,
            },
        },
    )
