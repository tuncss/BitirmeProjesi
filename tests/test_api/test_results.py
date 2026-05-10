"""Result polling endpoint tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


def test_results_not_found_returns_404(monkeypatch, api_client: TestClient) -> None:
    class PendingAsyncResult:
        def __init__(self, task_id: str, app: Any):
            self.state = "PENDING"
            self.info = None

    monkeypatch.setattr("backend.app.api.routes.results.AsyncResult", PendingAsyncResult)

    response = api_client.get("/api/results/unknown-task")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


def test_results_pending_uses_submitted_task_record(
    monkeypatch,
    api_client: TestClient,
    api_tmp_root: Path,
) -> None:
    class PendingAsyncResult:
        def __init__(self, task_id: str, app: Any):
            self.state = "PENDING"
            self.info = None

    monkeypatch.setattr("backend.app.api.routes.results.AsyncResult", PendingAsyncResult)
    result_dir = api_tmp_root / "results" / "task123"
    result_dir.mkdir(parents=True)
    (result_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "task123",
                "celery_task_id": "task123",
                "session_id": "session123",
                "model_name": "unet3d",
                "status": "submitted",
            }
        ),
        encoding="utf-8",
    )

    response = api_client.get("/api/results/task123")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "task123",
        "status": "pending",
        "progress": 0,
        "step": "queued",
        "message": "Segmentation task is queued or waiting to start.",
        "model_name": "unet3d",
        "celery_task_id": "task123",
    }


def test_results_completed_response_includes_validation(
    api_client: TestClient,
    api_tmp_root: Path,
) -> None:
    result_dir = api_tmp_root / "results" / "task_with_validation"
    result_dir.mkdir(parents=True)
    metadata = {
        "task_id": "task_with_validation",
        "model_name": "unet3d",
        "validation": {
            "subject_id": "BraTS2021_00000",
            "metrics": {
                "WT": {"dice": 0.9, "hd95": 1.0},
                "TC": {"dice": 0.8, "hd95": 2.0},
                "ET": {"dice": 0.7, "hd95": -1.0},
            },
            "gt_filename": "ground_truth.nii.gz",
        },
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    response = api_client.get("/api/results/task_with_validation")

    assert response.status_code == 200
    assert response.json()["results"]["validation"] == metadata["validation"]


def test_result_file_endpoint_returns_ground_truth_file(
    api_client: TestClient,
    api_tmp_root: Path,
) -> None:
    result_dir = api_tmp_root / "results" / "task_with_gt"
    result_dir.mkdir(parents=True)
    expected = b"ground-truth-bytes"
    (result_dir / "ground_truth.nii.gz").write_bytes(expected)

    response = api_client.get("/api/results/task_with_gt/files/ground_truth.nii.gz")

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"] == "application/octet-stream"


def test_result_file_endpoint_rejects_missing_and_non_whitelisted_files(
    api_client: TestClient,
    api_tmp_root: Path,
) -> None:
    result_dir = api_tmp_root / "results" / "task_with_files"
    result_dir.mkdir(parents=True)
    (result_dir / "metadata.json").write_text("{}", encoding="utf-8")

    missing_response = api_client.get("/api/results/task_with_files/files/ground_truth.nii.gz")
    forbidden_response = api_client.get("/api/results/task_with_files/files/metadata.json")
    traversal_response = api_client.get("/api/results/task_with_files/files/../metadata.json")

    assert missing_response.status_code == 404
    assert forbidden_response.status_code == 404
    assert traversal_response.status_code == 404


def test_results_failure_state_returns_error(monkeypatch, api_client: TestClient) -> None:
    class FailedAsyncResult:
        def __init__(self, task_id: str, app: Any):
            self.state = "FAILURE"
            self.info = {
                "step": "error",
                "progress": 100,
                "message": "Segmentation failed.",
                "error": "bad input",
            }

    monkeypatch.setattr("backend.app.api.routes.results.AsyncResult", FailedAsyncResult)

    response = api_client.get("/api/results/task456")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "task456",
        "status": "failed",
        "progress": 100,
        "step": "error",
        "message": "Segmentation failed.",
        "error": "bad input",
    }
