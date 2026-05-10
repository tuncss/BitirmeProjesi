"""Celery task that runs full brain tumor segmentation inference."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from loguru import logger

from backend.app.core.config import PROJECT_ROOT, Settings, get_settings
from backend.app.services.brats_lookup import extract_subject_id, resolve_gt_path
from backend.app.services.file_manager import FileManager
from backend.app.tasks.celery_app import celery_app
from src.data.utils import compute_voxel_volume, load_nifti, save_nifti
from src.inference.postprocessing import SegmentationPostProcessor
from src.inference.predictor import BrainTumorPredictor, compute_tumor_volumes
from src.inference.validation import compute_validation_metrics


SEGMENTATION_TASK_NAME = "segmentation.run"
SEGMENTATION_FILENAME = "segmentation.nii.gz"
METADATA_FILENAME = "metadata.json"
FAILURE_FILENAME = "failure.json"
BACKGROUND_FILENAME = "background.nii.gz"
GROUND_TRUTH_FILENAME = "ground_truth.nii.gz"
BACKGROUND_MODALITY = "flair"  # FLAIR best highlights peritumoral edema
SEGMENTATION_CLASSES = {
    "0": "Background",
    "1": "NCR/NET (Necrotic Tumor Core)",
    "2": "ED (Peritumoral Edema)",
    "4": "ET (Enhancing Tumor)",
}


class TaskStateUpdater(Protocol):
    """Small protocol for Celery tasks and test doubles that update progress."""

    def update_state(self, *, state: str | None = None, meta: dict[str, Any] | None = None) -> None:
        """Update task state in the result backend."""


_predictors: dict[tuple[str, str, str], BrainTumorPredictor] = {}


def get_predictor(model_name: str, settings: Settings | None = None) -> BrainTumorPredictor:
    """Lazy-load a predictor once per model/path/device in the worker process."""
    app_settings = settings or get_settings()
    model_path = app_settings.model_paths.get(model_name)
    if model_path is None:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = resolve_runtime_path(model_path)
    device = resolve_device(app_settings.segmentation_device)
    cache_key = (model_name, str(checkpoint_path), device)

    if cache_key not in _predictors:
        _predictors[cache_key] = BrainTumorPredictor(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info(f"Model loaded and cached: {model_name} ({checkpoint_path})")

    return _predictors[cache_key]


def execute_segmentation(
    session_id: str,
    model_name: str,
    task_id: str,
    *,
    settings: Settings | None = None,
    task_context: TaskStateUpdater | None = None,
    file_manager: FileManager | None = None,
    post_processor: SegmentationPostProcessor | None = None,
) -> dict[str, Any]:
    """Run one segmentation job and persist its NIfTI mask plus metadata."""
    app_settings = settings or get_settings()
    manager = file_manager or FileManager(app_settings)
    processor = post_processor or SegmentationPostProcessor()
    start_time = time.time()

    _update_progress(
        task_context,
        step="preprocessing",
        progress=10,
        message="Loading and validating NIfTI files.",
    )
    modality_map = manager.identify_modalities(session_id)
    manager.validate_nifti_files(modality_map)

    _update_progress(
        task_context,
        step="inference",
        progress=30,
        message=f"Running segmentation with {model_name}.",
    )
    predictor = get_predictor(model_name, app_settings)
    prediction = predictor.predict(modality_map)

    _update_progress(
        task_context,
        step="postprocessing",
        progress=80,
        message="Cleaning segmentation mask.",
    )
    cleaned_mask = processor.process(prediction["segmentation_mask"])
    affine = np.asarray(prediction["affine"])
    tumor_volumes = compute_tumor_volumes(cleaned_mask, compute_voxel_volume(affine))

    result_dir = manager.create_result_dir(task_id)
    segmentation_path = result_dir / SEGMENTATION_FILENAME
    metadata_path = result_dir / METADATA_FILENAME

    _update_progress(
        task_context,
        step="validation",
        progress=90,
        message="Checking BraTS ground-truth validation data.",
    )
    validation = _compute_validation_result(
        manager=manager,
        session_id=session_id,
        cleaned_mask=cleaned_mask,
        result_dir=result_dir,
        raw_root=app_settings.brats_raw_root,
    )

    save_nifti(cleaned_mask.astype(np.int16, copy=False), affine, segmentation_path)

    background_source = modality_map.get(BACKGROUND_MODALITY)
    background_path: Path | None = None
    if background_source is not None and Path(background_source).exists():
        background_path = result_dir / BACKGROUND_FILENAME
        shutil.copyfile(background_source, background_path)

    elapsed_time = time.time() - start_time
    metadata = build_metadata(
        task_id=task_id,
        model_name=model_name,
        processing_time_seconds=elapsed_time,
        tumor_volumes=tumor_volumes,
        original_shape=prediction["original_shape"],
        result_dir=result_dir,
        segmentation_path=segmentation_path,
        crop_bbox=prediction.get("crop_bbox"),
        background_path=background_path,
        validation=validation,
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    _update_progress(
        task_context,
        step="completed",
        progress=100,
        message="Segmentation completed.",
    )
    logger.info(f"Segmentation complete: {task_id} ({elapsed_time:.1f}s)")

    return {
        "status": "completed",
        "task_id": task_id,
        "model_name": model_name,
        "processing_time_seconds": round(elapsed_time, 2),
        "tumor_volumes": tumor_volumes,
        "result_dir": str(result_dir),
        "segmentation_path": str(segmentation_path),
        "metadata_path": str(metadata_path),
    }


@celery_app.task(bind=True, name=SEGMENTATION_TASK_NAME)
def run_segmentation(self: TaskStateUpdater, session_id: str, model_name: str, task_id: str) -> dict[str, Any]:
    """Celery entry point for asynchronous segmentation."""
    try:
        return execute_segmentation(
            session_id=session_id,
            model_name=model_name,
            task_id=task_id,
            task_context=self,
        )
    except Exception as exc:
        logger.exception(f"Segmentation failed: {task_id} - {exc}")
        # Persist failure details to disk so the API can serve them even after
        # Celery overwrites task meta with its own exception envelope on raise.
        _persist_failure(task_id=task_id, exc=exc)
        raise


def _persist_failure(*, task_id: str, exc: BaseException) -> None:
    """Write a failure record next to the would-be result artifacts."""
    try:
        manager = FileManager()
        result_dir = manager.create_result_dir(task_id)
        record = {
            "task_id": task_id,
            "status": "failed",
            "step": "error",
            "progress": 100,
            "message": str(exc),
            "error": f"{type(exc).__name__}: {exc}",
        }
        with (result_dir / FAILURE_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
    except Exception:  # noqa: BLE001 - never let bookkeeping mask the real failure
        logger.exception(f"Could not persist failure record for task {task_id}")


def build_metadata(
    *,
    task_id: str,
    model_name: str,
    processing_time_seconds: float,
    tumor_volumes: dict[str, float],
    original_shape: tuple[int, ...] | list[int],
    result_dir: Path,
    segmentation_path: Path,
    crop_bbox: Any | None = None,
    background_path: Path | None = None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON metadata written beside the segmentation mask."""
    files: dict[str, str] = {
        "segmentation": segmentation_path.name,
        "metadata": METADATA_FILENAME,
        "result_dir": str(result_dir),
    }
    if background_path is not None:
        files["background"] = background_path.name

    metadata: dict[str, Any] = {
        "task_id": task_id,
        "model_name": model_name,
        "processing_time_seconds": round(float(processing_time_seconds), 2),
        "tumor_volumes": {key: float(value) for key, value in tumor_volumes.items()},
        "original_shape": [int(value) for value in original_shape],
        "segmentation_classes": SEGMENTATION_CLASSES,
        "has_background": background_path is not None,
        "files": files,
    }
    if validation is not None:
        files["ground_truth"] = validation["gt_filename"]
        metadata["validation"] = validation

    serialized_bbox = serialize_crop_bbox(crop_bbox)
    if serialized_bbox is not None:
        metadata["crop_bbox"] = serialized_bbox

    return metadata


def _compute_validation_result(
    *,
    manager: FileManager,
    session_id: str,
    cleaned_mask: np.ndarray,
    result_dir: Path,
    raw_root: Path,
) -> dict[str, Any] | None:
    """Compute optional BraTS validation metrics without failing inference."""
    try:
        original_filenames = manager.get_original_filenames(session_id)
        subject_id = extract_subject_id(original_filenames)
        if subject_id is None:
            return None

        gt_path = resolve_gt_path(subject_id, raw_root)
        if gt_path is None:
            return None

        gt_array, gt_affine = load_nifti(gt_path)
        if tuple(gt_array.shape) != tuple(cleaned_mask.shape):
            logger.warning(
                f"Skipping validation for {subject_id} because GT shape "
                f"{gt_array.shape} does not match prediction shape {cleaned_mask.shape}"
            )
            return None

        spacing = compute_voxel_spacing(gt_affine)
        metrics = compute_validation_metrics(cleaned_mask, gt_array, spacing=spacing)
        shutil.copyfile(gt_path, result_dir / GROUND_TRUTH_FILENAME)
        return {
            "subject_id": subject_id,
            "metrics": metrics,
            "gt_filename": GROUND_TRUTH_FILENAME,
        }
    except Exception as exc:  # noqa: BLE001 - validation must not fail segmentation
        logger.warning(f"Skipping validation for session {session_id}: {exc}")
        return None


def compute_voxel_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    """Compute voxel spacing from a NIfTI affine matrix."""
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError(f"Affine must be at least 3x3, got {matrix.shape}")
    return tuple(float(value) for value in np.linalg.norm(matrix[:3, :3], axis=0))


def serialize_crop_bbox(crop_bbox: Any | None) -> dict[str, list[int]] | None:
    """Return a JSON-safe crop bbox representation when predictor metadata exists."""
    if crop_bbox is None:
        return None
    if not all(hasattr(crop_bbox, name) for name in ("starts", "stops", "shape")):
        return None
    return {
        "starts": [int(value) for value in crop_bbox.starts],
        "stops": [int(value) for value in crop_bbox.stops],
        "shape": [int(value) for value in crop_bbox.shape],
    }


def resolve_runtime_path(path: str | Path) -> Path:
    """Resolve model/config paths relative to the project root when needed."""
    runtime_path = Path(path).expanduser()
    if runtime_path.is_absolute():
        return runtime_path
    return PROJECT_ROOT / runtime_path


def resolve_device(configured_device: str) -> str:
    """Resolve segmentation device to a concrete string ('cuda'/'cpu')."""
    import torch

    device = configured_device.strip().lower()
    cuda_available = torch.cuda.is_available()
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    if device.startswith("cuda") and not cuda_available:
        logger.warning(
            f"Configured device '{device}' but CUDA is unavailable; falling back to 'cpu'."
        )
        return "cpu"
    return device


def _update_progress(
    task_context: TaskStateUpdater | None,
    *,
    step: str,
    progress: int,
    message: str,
    state: str = "PROCESSING",
    error: str | None = None,
) -> None:
    if task_context is None:
        return

    meta: dict[str, Any] = {
        "step": step,
        "progress": progress,
        "message": message,
    }
    if error is not None:
        meta["error"] = error

    task_context.update_state(state=state, meta=meta)
