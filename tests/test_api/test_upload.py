"""Upload endpoint success and failure tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from tests.test_api.conftest import make_nifti_bytes


def test_upload_invalid_format_returns_consistent_400(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/upload",
        files=[("files", ("notes.txt", b"not a nifti", "text/plain"))],
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_FILE_FORMAT"


def test_upload_file_too_large_returns_413(api_tmp_root: Path) -> None:
    app = create_app(
        Settings(
            app_env="test",
            upload_dir=api_tmp_root / "uploads",
            results_dir=api_tmp_root / "results",
            max_upload_size_mb=0,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/upload",
        files=[("files", ("case_t1.nii.gz", b"x", "application/gzip"))],
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_upload_zip_extracts_modalities(api_client: TestClient, api_tmp_root: Path) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        for modality in ["t1", "t1ce", "t2", "flair"]:
            archive.writestr(
                f"nested/BraTS2021_00000_{modality}.nii.gz",
                make_nifti_bytes(api_tmp_root, modality),
            )

    response = api_client.post(
        "/api/upload",
        files=[("files", ("modalities.zip", archive_bytes.getvalue(), "application/zip"))],
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "uploaded"
    assert sorted(payload["extracted_files"]) == [
        "BraTS2021_00000_flair.nii.gz",
        "BraTS2021_00000_t1.nii.gz",
        "BraTS2021_00000_t1ce.nii.gz",
        "BraTS2021_00000_t2.nii.gz",
    ]
    assert set(payload["modalities"]) == {"t1", "t1ce", "t2", "flair"}

