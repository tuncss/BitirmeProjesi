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

