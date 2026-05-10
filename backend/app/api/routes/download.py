"""Download endpoint for segmentation masks and the background MR volume.

Two query modes:
  - ?type=segmentation|background  (default: served inline as octet-stream so
    Niivue's fetch can read it; download manager extensions like IDM ignore it)
  - &disposition=attachment        (forces a download response — used by the
    explicit "İndir" button in the UI)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from backend.app.api.dependencies import get_app_settings
from backend.app.core.exceptions import TaskNotFoundError


router = APIRouter(tags=["download"])


DownloadType = Literal["segmentation", "background", "ground_truth"]
Disposition = Literal["inline", "attachment"]
_FILES: dict[DownloadType, str] = {
    "segmentation": "segmentation.nii.gz",
    "background": "background.nii.gz",
    "ground_truth": "ground_truth.nii.gz",
}


@router.get("/download/{task_id}", summary="Download segmentation mask or background MR")
async def download_result(
    request: Request,
    task_id: str,
    type: DownloadType = Query("segmentation", description="segmentation | background"),
    disposition: Disposition = Query(
        "inline",
        description="inline (viewer fetch) | attachment (force download)",
    ),
) -> FileResponse:
    """Stream a saved task artifact (segmentation mask or background MR)."""
    settings = get_app_settings(request)
    filename = _FILES[type]
    result_path = Path(settings.results_dir) / task_id / filename
    if not result_path.exists():
        raise TaskNotFoundError(task_id)

    if disposition == "attachment":
        return FileResponse(
            path=str(result_path),
            filename=f"{type}_{task_id}.nii.gz",
            media_type="application/gzip",
        )

    # Inline serving for in-app viewers: octet-stream + no attachment header
    # so download managers (IDM, FDM, etc.) leave the response alone.
    return FileResponse(
        path=str(result_path),
        media_type="application/octet-stream",
    )
