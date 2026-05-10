"""Segmentation submission endpoint tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient


def test_segment_invalid_model_returns_400(api_client: TestClient, valid_upload_session: str) -> None:
    response = api_client.post(
        "/api/segment",
        json={"session_id": valid_upload_session, "model_name": "bad_model"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "INVALID_MODEL"
    assert payload["error"]["details"]["allowed_models"] == ["attention_unet3d", "unet3d"]


def test_segment_missing_session_returns_404(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/segment",
        json={"session_id": "missing-session", "model_name": "unet3d"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "FILE_NOT_FOUND"


def test_segment_queue_failure_returns_503(
    monkeypatch,
    api_client: TestClient,
    valid_upload_session: str,
) -> None:
    class FailingRunSegmentation:
        @staticmethod
        def apply_async(*, kwargs: dict[str, Any], task_id: str):
            raise RuntimeError("Redis unavailable")

    monkeypatch.setattr("backend.app.api.routes.segment.run_segmentation", FailingRunSegmentation)

    response = api_client.post(
        "/api/segment",
        json={"session_id": valid_upload_session, "model_name": "unet3d"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "TASK_QUEUE_UNAVAILABLE"


def test_segment_submission_writes_task_record(
    monkeypatch,
    api_client: TestClient,
    api_tmp_root: Path,
    valid_upload_session: str,
) -> None:
    class FakeRunSegmentation:
        @staticmethod
        def apply_async(*, kwargs: dict[str, Any], task_id: str):
            return SimpleNamespace(id=task_id)

    monkeypatch.setattr("backend.app.api.routes.segment.run_segmentation", FakeRunSegmentation)

    response = api_client.post(
        "/api/segment",
        json={"session_id": valid_upload_session, "model_name": "unet3d"},
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    task_record = json.loads((api_tmp_root / "results" / task_id / "task.json").read_text(encoding="utf-8"))
    assert task_record == {
        "task_id": task_id,
        "celery_task_id": task_id,
        "session_id": valid_upload_session,
        "model_name": "unet3d",
        "status": "submitted",
    }

