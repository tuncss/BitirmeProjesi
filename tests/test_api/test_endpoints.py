"""Tests for FastAPI upload, task submission, results, and download endpoints."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_upload_endpoint_accepts_four_nifti_modalities() -> None:
    root = _fresh_tmp_dir("test_endpoint_upload")
    try:
        client = _client(root)
        files = [
            ("files", (f"BraTS2021_00000_{modality}.nii.gz", _nifti_bytes(root, modality), "application/gzip"))
            for modality in ["t1", "t1ce", "t2", "flair"]
        ]

        response = client.post("/api/upload", files=files)

        assert response.status_code == 201
        payload = response.json()
        assert payload["status"] == "uploaded"
        assert set(payload["modalities"]) == {"t1", "t1ce", "t2", "flair"}
        assert (root / "uploads" / payload["session_id"]).exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_upload_endpoint_rejects_incomplete_modalities() -> None:
    root = _fresh_tmp_dir("test_endpoint_upload_missing")
    try:
        client = _client(root)
        files = [
            ("files", ("case_t1.nii.gz", _nifti_bytes(root, "t1"), "application/gzip")),
        ]

        response = client.post("/api/upload", files=files)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_UPLOAD"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_segment_endpoint_submits_celery_task(monkeypatch) -> None:
    root = _fresh_tmp_dir("test_endpoint_segment")
    captured: dict[str, Any] = {}

    class FakeRunSegmentation:
        @staticmethod
        def apply_async(*, kwargs: dict[str, Any], task_id: str):
            captured["kwargs"] = kwargs
            captured["task_id"] = task_id
            return SimpleNamespace(id=task_id)

    monkeypatch.setattr("backend.app.api.routes.segment.run_segmentation", FakeRunSegmentation)

    try:
        client = _client(root)
        session_id = _create_valid_upload_session(root)

        response = client.post(
            "/api/segment",
            json={"session_id": session_id, "model_name": "unet3d"},
        )

        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "submitted"
        assert payload["model_name"] == "unet3d"
        assert payload["task_id"] == payload["celery_task_id"]
        assert captured["task_id"] == payload["task_id"]
        assert captured["kwargs"] == {
            "session_id": session_id,
            "model_name": "unet3d",
            "task_id": payload["task_id"],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_segment_endpoint_uses_default_model(monkeypatch) -> None:
    root = _fresh_tmp_dir("test_endpoint_segment_default")

    class FakeRunSegmentation:
        @staticmethod
        def apply_async(*, kwargs: dict[str, Any], task_id: str):
            return SimpleNamespace(id=task_id, kwargs=kwargs)

    monkeypatch.setattr("backend.app.api.routes.segment.run_segmentation", FakeRunSegmentation)

    try:
        client = _client(root)
        session_id = _create_valid_upload_session(root)

        response = client.post("/api/segment", json={"session_id": session_id})

        assert response.status_code == 202
        assert response.json()["model_name"] == "attention_unet3d"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_results_endpoint_returns_metadata_when_complete() -> None:
    root = _fresh_tmp_dir("test_endpoint_results_complete")
    try:
        client = _client(root)
        result_dir = root / "results" / "task123"
        result_dir.mkdir(parents=True)
        metadata = {
            "task_id": "task123",
            "model_name": "unet3d",
            "tumor_volumes": {"WT_cm3": 1.0, "TC_cm3": 0.5, "ET_cm3": 0.25},
        }
        (result_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        response = client.get("/api/results/task123")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "completed"
        assert payload["progress"] == 100
        assert payload["results"] == metadata
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_results_endpoint_returns_processing_state(monkeypatch) -> None:
    root = _fresh_tmp_dir("test_endpoint_results_processing")

    class FakeAsyncResult:
        def __init__(self, task_id: str, app: Any):
            self.state = "PROCESSING"
            self.info = {"step": "inference", "progress": 30, "message": "Running."}

    monkeypatch.setattr("backend.app.api.routes.results.AsyncResult", FakeAsyncResult)

    try:
        client = _client(root)

        response = client.get("/api/results/task456")

        assert response.status_code == 200
        assert response.json() == {
            "task_id": "task456",
            "status": "processing",
            "progress": 30,
            "step": "inference",
            "message": "Running.",
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_download_endpoint_returns_segmentation_file() -> None:
    root = _fresh_tmp_dir("test_endpoint_download")
    try:
        client = _client(root)
        result_dir = root / "results" / "task789"
        result_dir.mkdir(parents=True)
        expected = b"nifti-bytes"
        (result_dir / "segmentation.nii.gz").write_bytes(expected)

        response = client.get("/api/download/task789")

        assert response.status_code == 200
        assert response.content == expected
        assert response.headers["content-type"] == "application/octet-stream"
        assert "content-disposition" not in response.headers

        attachment_response = client.get("/api/download/task789?disposition=attachment")
        assert attachment_response.status_code == 200
        assert attachment_response.content == expected
        assert attachment_response.headers["content-type"] == "application/gzip"
        assert "segmentation_task789.nii.gz" in attachment_response.headers["content-disposition"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_download_endpoint_returns_ground_truth_file() -> None:
    root = _fresh_tmp_dir("test_endpoint_download_gt")
    try:
        client = _client(root)
        result_dir = root / "results" / "task_gt"
        result_dir.mkdir(parents=True)
        expected = b"ground-truth-bytes"
        (result_dir / "ground_truth.nii.gz").write_bytes(expected)

        response = client.get("/api/download/task_gt?type=ground_truth")

        assert response.status_code == 200
        assert response.content == expected
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_download_endpoint_returns_consistent_404_for_missing_task() -> None:
    root = _fresh_tmp_dir("test_endpoint_download_missing")
    try:
        client = _client(root)

        response = client.get("/api/download/missing")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "TASK_NOT_FOUND"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _client(root: Path) -> TestClient:
    app = create_app(
        Settings(
            app_env="test",
            upload_dir=root / "uploads",
            results_dir=root / "results",
            default_model_name="attention_unet3d",
            unet_model_path=root / "unet.pth",
            attention_unet_model_path=root / "attention.pth",
        )
    )
    return TestClient(app)


def _create_valid_upload_session(root: Path) -> str:
    session_id = uuid.uuid4().hex
    session_dir = root / "uploads" / session_id
    session_dir.mkdir(parents=True)
    for modality in ["t1", "t1ce", "t2", "flair"]:
        (session_dir / f"BraTS2021_00000_{modality}.nii.gz").write_bytes(_nifti_bytes(root, modality))
    return session_id


def _nifti_bytes(root: Path, name: str) -> bytes:
    path = root / f"{name}_{uuid.uuid4().hex}.nii.gz"
    data = np.zeros((4, 4, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    content = path.read_bytes()
    path.unlink()
    return content


def _fresh_tmp_dir(prefix: str) -> Path:
    root = Path("tmp") / f"{prefix}_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
