"""Slow backend integration tests for segmentation validation."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import PROJECT_ROOT, Settings
from backend.app.main import create_app
from backend.app.tasks import segmentation_task


SUBJECT_ID = "BraTS2021_00000"
MODALITIES = ("t1", "t1ce", "t2", "flair")


@pytest.mark.slow
def test_validation_e2e_brats_upload_segment_results_and_gt_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fresh_tmp_dir("test_validation_e2e")
    try:
        settings = _settings(root)
        client = TestClient(create_app(settings))
        _patch_segmentation_to_run_inline(monkeypatch, settings)
        _patch_predictor_to_return_gt(monkeypatch, settings.brats_raw_root, SUBJECT_ID)

        upload_response = client.post(
            "/api/upload",
            files=_upload_files(SUBJECT_ID),
        )
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]

        segment_response = client.post(
            "/api/segment",
            json={"session_id": session_id, "model_name": "unet3d"},
        )
        assert segment_response.status_code == 202
        task_id = segment_response.json()["task_id"]

        results_response = client.get(f"/api/results/{task_id}")
        assert results_response.status_code == 200
        payload = results_response.json()
        validation = payload["results"]["validation"]

        assert payload["status"] == "completed"
        assert validation["subject_id"] == SUBJECT_ID
        assert validation["metrics"]["WT"]["dice"] >= 0.85
        assert validation["gt_filename"] == "ground_truth.nii.gz"
        assert (settings.results_dir / task_id / "metadata.json").exists()
        assert (settings.results_dir / task_id / "ground_truth.nii.gz").exists()

        gt_response = client.get(f"/api/results/{task_id}/files/ground_truth.nii.gz")
        assert gt_response.status_code == 200
        assert gt_response.content.startswith(b"\x1f\x8b")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.slow
def test_validation_e2e_non_brats_names_complete_without_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fresh_tmp_dir("test_validation_e2e_non_brats")
    try:
        settings = _settings(root)
        client = TestClient(create_app(settings))
        _patch_segmentation_to_run_inline(monkeypatch, settings)
        _patch_predictor_to_return_gt(monkeypatch, settings.brats_raw_root, SUBJECT_ID)

        upload_response = client.post(
            "/api/upload",
            files=_upload_files(SUBJECT_ID, filename_prefix="patient_a"),
        )
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]

        segment_response = client.post(
            "/api/segment",
            json={"session_id": session_id, "model_name": "unet3d"},
        )
        assert segment_response.status_code == 202
        task_id = segment_response.json()["task_id"]

        results_response = client.get(f"/api/results/{task_id}")
        assert results_response.status_code == 200
        metadata = results_response.json()["results"]

        assert results_response.json()["status"] == "completed"
        assert "validation" not in metadata
        assert not (settings.results_dir / task_id / "ground_truth.nii.gz").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


class GroundTruthPredictor:
    def __init__(self, raw_root: Path, subject_id: str):
        gt_path = raw_root / "BraTS2021" / subject_id / f"{subject_id}_seg.nii.gz"
        image = nib.load(str(gt_path))
        self.mask = np.asarray(image.get_fdata(), dtype=np.int16)
        self.affine = image.affine

    def predict(self, modality_paths: dict[str, Path]) -> dict[str, Any]:
        assert set(modality_paths) == set(MODALITIES)
        return {
            "segmentation_mask": self.mask,
            "affine": self.affine,
            "original_shape": tuple(int(value) for value in self.mask.shape),
            "tumor_volumes": {},
            "model_name": "unet3d",
        }


def _settings(root: Path) -> Settings:
    raw_root = PROJECT_ROOT / "data" / "raw"
    _require_brats_case(raw_root, SUBJECT_ID)
    return Settings(
        app_env="test",
        upload_dir=root / "uploads",
        results_dir=root / "results",
        brats_raw_root=raw_root,
        default_model_name="unet3d",
        unet_model_path=root / "unet.pth",
        attention_unet_model_path=root / "attention.pth",
    )


def _patch_segmentation_to_run_inline(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    class InlineRunSegmentation:
        @staticmethod
        def apply_async(*, kwargs: dict[str, Any], task_id: str):
            segmentation_task.execute_segmentation(
                session_id=kwargs["session_id"],
                model_name=kwargs["model_name"],
                task_id=kwargs["task_id"],
                settings=settings,
            )
            return SimpleNamespace(id=task_id)

    monkeypatch.setattr("backend.app.api.routes.segment.run_segmentation", InlineRunSegmentation)


def _patch_predictor_to_return_gt(
    monkeypatch: pytest.MonkeyPatch,
    raw_root: Path,
    subject_id: str,
) -> None:
    predictor = GroundTruthPredictor(raw_root, subject_id)
    monkeypatch.setattr(
        "backend.app.tasks.segmentation_task.get_predictor",
        lambda model_name, settings=None: predictor,
    )


def _upload_files(subject_id: str, filename_prefix: str | None = None) -> list[tuple[str, tuple[str, bytes, str]]]:
    case_dir = PROJECT_ROOT / "data" / "raw" / "BraTS2021" / subject_id
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for modality in MODALITIES:
        source = case_dir / f"{subject_id}_{modality}.nii.gz"
        filename = f"{filename_prefix}_{modality}.nii.gz" if filename_prefix else source.name
        files.append(("files", (filename, source.read_bytes(), "application/gzip")))
    return files


def _require_brats_case(raw_root: Path, subject_id: str) -> None:
    case_dir = raw_root / "BraTS2021" / subject_id
    required = [case_dir / f"{subject_id}_{modality}.nii.gz" for modality in MODALITIES]
    required.append(case_dir / f"{subject_id}_seg.nii.gz")
    missing = [path for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"BraTS case {subject_id} is not available under {raw_root}")


def _fresh_tmp_dir(prefix: str) -> Path:
    root = Path("tmp") / f"{prefix}_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
