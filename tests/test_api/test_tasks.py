"""Tests for Celery segmentation task orchestration."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from backend.app.core.config import PROJECT_ROOT, Settings
from backend.app.services import FileManager
from backend.app.tasks.celery_app import celery_app
from backend.app.tasks.segmentation_task import (
    SEGMENTATION_TASK_NAME,
    _predictors,
    build_metadata,
    execute_segmentation,
    get_predictor,
    resolve_device,
    resolve_runtime_path,
)
from src.inference import SegmentationPostProcessor


def test_celery_app_uses_redis_settings_and_registers_segmentation_task() -> None:
    assert celery_app.conf.broker_url == "redis://localhost:6379/0"
    assert celery_app.conf.result_backend == "redis://localhost:6379/0"
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert SEGMENTATION_TASK_NAME in celery_app.tasks


def test_runtime_helpers_resolve_paths_and_auto_device() -> None:
    assert resolve_runtime_path("data/models/model.pth") == PROJECT_ROOT / "data/models/model.pth"
    assert resolve_runtime_path(PROJECT_ROOT / "model.pth") == PROJECT_ROOT / "model.pth"
    assert resolve_device("auto") is None
    assert resolve_device("cpu") == "cpu"


def test_get_predictor_lazily_loads_and_caches_by_model_path_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fresh_tmp_dir("test_task_predictor_cache")
    _predictors.clear()
    created: list[FakePredictor] = []

    class RecordingPredictor(FakePredictor):
        def __init__(self, model_name: str, checkpoint_path: str | Path, device: str | None = None):
            super().__init__(model_name=model_name, checkpoint_path=checkpoint_path, device=device)
            created.append(self)

    monkeypatch.setattr("backend.app.tasks.segmentation_task.BrainTumorPredictor", RecordingPredictor)

    try:
        settings = Settings(
            unet_model_path=root / "unet.pth",
            attention_unet_model_path=root / "attention.pth",
            segmentation_device="cpu",
        )

        first = get_predictor("unet3d", settings)
        second = get_predictor("unet3d", settings)
        attention = get_predictor("attention_unet3d", settings)

        assert first is second
        assert first is created[0]
        assert attention is created[1]
        assert len(created) == 2
        assert created[0].model_name == "unet3d"
        assert created[0].checkpoint_path == resolve_runtime_path(root / "unet.pth")
        assert created[0].device == "cpu"
    finally:
        _predictors.clear()
        shutil.rmtree(root, ignore_errors=True)


def test_get_predictor_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_predictor("bad_model", Settings())


def test_execute_segmentation_writes_mask_metadata_and_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fresh_tmp_dir("test_task_execute")
    manager = _manager(root)
    task_context = RecordingTaskContext()

    mask = np.zeros((4, 4, 4), dtype=np.int16)
    mask[0, 0, 0] = 4
    mask[0, 0, 1] = 4
    mask[1, 0, 0] = 1
    mask[2, 0, 0] = 2
    predictor = FakePredictor(mask=mask, affine=np.diag([10, 10, 10, 1]))

    monkeypatch.setattr("backend.app.tasks.segmentation_task.get_predictor", lambda model_name, settings=None: predictor)

    try:
        session_id = _write_upload_session(manager)
        result = execute_segmentation(
            session_id=session_id,
            model_name="attention_unet3d",
            task_id="task123",
            settings=manager.settings,
            task_context=task_context,
            file_manager=manager,
            post_processor=SegmentationPostProcessor(min_component_size=0, fill_holes=False),
        )

        result_dir = manager.results_dir / "task123"
        metadata_path = result_dir / "metadata.json"
        segmentation_path = result_dir / "segmentation.nii.gz"

        assert result["status"] == "completed"
        assert result["model_name"] == "attention_unet3d"
        assert result["tumor_volumes"] == {"WT_cm3": 4.0, "TC_cm3": 3.0, "ET_cm3": 2.0}
        assert segmentation_path.exists()
        assert metadata_path.exists()

        saved_mask = np.asarray(nib.load(str(segmentation_path)).get_fdata(), dtype=np.int16)
        assert np.array_equal(saved_mask, mask)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["task_id"] == "task123"
        assert metadata["model_name"] == "attention_unet3d"
        assert metadata["tumor_volumes"] == {"WT_cm3": 4.0, "TC_cm3": 3.0, "ET_cm3": 2.0}
        assert metadata["original_shape"] == [4, 4, 4]
        assert metadata["files"]["segmentation"] == "segmentation.nii.gz"
        assert [entry["meta"]["step"] for entry in task_context.states] == [
            "preprocessing",
            "inference",
            "postprocessing",
            "completed",
        ]
        assert task_context.states[-1]["meta"]["progress"] == 100
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_execute_segmentation_applies_postprocessing_before_saving(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fresh_tmp_dir("test_task_postprocessing")
    manager = _manager(root)
    mask = np.zeros((4, 4, 4), dtype=np.int16)
    mask[1, 1, 1] = 4
    predictor = FakePredictor(mask=mask, affine=np.diag([10, 10, 10, 1]))

    monkeypatch.setattr("backend.app.tasks.segmentation_task.get_predictor", lambda model_name, settings=None: predictor)

    try:
        session_id = _write_upload_session(manager)
        result = execute_segmentation(
            session_id=session_id,
            model_name="unet3d",
            task_id="task456",
            settings=manager.settings,
            file_manager=manager,
            post_processor=SegmentationPostProcessor(min_component_size=2, fill_holes=False),
        )

        saved_mask = np.asarray(nib.load(str(manager.results_dir / "task456" / "segmentation.nii.gz")).get_fdata())
        metadata = json.loads((manager.results_dir / "task456" / "metadata.json").read_text(encoding="utf-8"))

        assert not saved_mask.any()
        assert result["tumor_volumes"] == {"WT_cm3": 0.0, "TC_cm3": 0.0, "ET_cm3": 0.0}
        assert metadata["tumor_volumes"] == {"WT_cm3": 0.0, "TC_cm3": 0.0, "ET_cm3": 0.0}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_build_metadata_omits_unknown_bbox_objects() -> None:
    metadata = build_metadata(
        task_id="task789",
        model_name="unet3d",
        processing_time_seconds=1.234,
        tumor_volumes={"WT_cm3": 1, "TC_cm3": 2, "ET_cm3": 3},
        original_shape=(1, 2, 3),
        result_dir=Path("results/task789"),
        segmentation_path=Path("results/task789/segmentation.nii.gz"),
        crop_bbox=object(),
    )

    assert "crop_bbox" not in metadata
    assert metadata["processing_time_seconds"] == 1.23
    assert metadata["tumor_volumes"] == {"WT_cm3": 1.0, "TC_cm3": 2.0, "ET_cm3": 3.0}


class RecordingTaskContext:
    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []

    def update_state(self, *, state: str | None = None, meta: dict[str, Any] | None = None) -> None:
        self.states.append({"state": state, "meta": meta or {}})


class FakePredictor:
    def __init__(
        self,
        model_name: str = "unet3d",
        checkpoint_path: str | Path = "model.pth",
        device: str | None = None,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.mask = mask if mask is not None else np.zeros((4, 4, 4), dtype=np.int16)
        self.affine = affine if affine is not None else np.eye(4)

    def predict(self, modality_paths: dict[str, Path]) -> dict[str, Any]:
        assert set(modality_paths) == {"t1", "t1ce", "t2", "flair"}
        return {
            "segmentation_mask": self.mask,
            "affine": self.affine,
            "original_shape": tuple(int(value) for value in self.mask.shape),
            "tumor_volumes": {},
            "model_name": self.model_name,
        }


def _manager(root: Path) -> FileManager:
    return FileManager(
        Settings(
            app_env="test",
            upload_dir=root / "uploads",
            results_dir=root / "results",
            allowed_extensions=(".nii", ".nii.gz", ".zip"),
            unet_model_path=root / "unet.pth",
            attention_unet_model_path=root / "attention.pth",
        )
    )


def _write_upload_session(manager: FileManager) -> str:
    session_id = manager.create_upload_session()
    session_dir = manager.get_upload_session_dir(session_id)
    for modality in ["t1", "t1ce", "t2", "flair"]:
        _write_nifti(session_dir / f"BraTS2021_00000_{modality}.nii.gz")
    return session_id


def _write_nifti(path: Path) -> None:
    data = np.zeros((4, 4, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _fresh_tmp_dir(prefix: str) -> Path:
    root = Path("tmp") / f"{prefix}_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
