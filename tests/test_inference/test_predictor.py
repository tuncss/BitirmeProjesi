"""Tests for inference predictor utilities."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src.inference import (
    BrainTumorPredictor,
    CropBoundingBox,
    InferenceMetadata,
    compute_nonzero_bbox,
    compute_tumor_volumes,
)


def test_compute_nonzero_bbox_uses_shared_modalities_and_margin() -> None:
    first = np.zeros((10, 10, 10), dtype=np.float32)
    second = np.zeros_like(first)
    first[3:5, 4:7, 5:8] = 1.0
    second[2, 2, 2] = 1.0

    bbox = compute_nonzero_bbox([first, second], margin=1)

    assert bbox.starts == (1, 1, 1)
    assert bbox.stops == (6, 8, 9)
    assert bbox.shape == (5, 7, 8)


def test_compute_nonzero_bbox_can_return_full_volume() -> None:
    volume = np.zeros((4, 5, 6), dtype=np.float32)

    bbox = compute_nonzero_bbox([volume], margin=1, enabled=False)

    assert bbox.starts == (0, 0, 0)
    assert bbox.stops == (4, 5, 6)
    assert bbox.shape == (4, 5, 6)


def test_compute_tumor_volumes_uses_brats_regions() -> None:
    mask = np.array([0, 1, 2, 4, 4])

    volumes = compute_tumor_volumes(mask, voxel_volume_mm3=2.0)

    assert volumes == {
        "WT_cm3": 0.01,
        "TC_cm3": 0.01,
        "ET_cm3": 0.0,
    }


def test_preprocess_modalities_validates_required_modalities(monkeypatch) -> None:
    monkeypatch.setattr(BrainTumorPredictor, "_load_model", lambda self, *_: _DummyModel())
    predictor = BrainTumorPredictor(
        "unet3d",
        "unused.pth",
        device="cpu",
        config=_small_config(),
    )

    with pytest.raises(ValueError, match="Missing modalities"):
        predictor.preprocess_modalities({"t1": "missing.nii.gz"})


def test_preprocess_modalities_rejects_shape_mismatch(monkeypatch) -> None:
    root = _fresh_tmp_dir()
    try:
        paths = _write_modalities(root, shapes={"flair": (5, 4, 4)})
        monkeypatch.setattr(BrainTumorPredictor, "_load_model", lambda self, *_: _DummyModel())
        predictor = BrainTumorPredictor(
            "unet3d",
            "unused.pth",
            device="cpu",
            config=_small_config(),
        )

        with pytest.raises(ValueError, match="same shape"):
            predictor.preprocess_modalities(paths)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_predict_runs_full_pipeline_with_dummy_model(monkeypatch) -> None:
    root = _fresh_tmp_dir()
    try:
        paths = _write_modalities(root)
        monkeypatch.setattr(BrainTumorPredictor, "_load_model", lambda self, *_: _DummyModel())
        predictor = BrainTumorPredictor(
            "unet3d",
            "unused.pth",
            device="cpu",
            config=_small_config(),
        )

        result = predictor.predict(paths)

        mask = result["segmentation_mask"]
        assert mask.shape == (4, 4, 4)
        assert set(np.unique(mask).tolist()).issubset({0, 4})
        assert result["probabilities"].shape == (4, 8, 8, 8)
        assert result["original_shape"] == (4, 4, 4)
        assert result["tumor_volumes"]["ET_cm3"] > 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_restore_to_original_places_prediction_inside_crop(monkeypatch) -> None:
    monkeypatch.setattr(BrainTumorPredictor, "_load_model", lambda self, *_: _DummyModel())
    predictor = BrainTumorPredictor(
        "unet3d",
        "unused.pth",
        device="cpu",
        config=_small_config(),
    )
    prediction = np.ones((8, 8, 8), dtype=np.int64)
    metadata = InferenceMetadata(
        affine=np.eye(4),
        original_shape=(6, 6, 6),
        crop_bbox=CropBoundingBox(starts=(1, 1, 1), stops=(5, 5, 5), shape=(4, 4, 4)),
        voxel_volume_mm3=1.0,
    )

    restored = predictor.restore_to_original(prediction, metadata)

    assert restored.shape == (6, 6, 6)
    assert restored[0].sum() == 0
    assert restored[:, 0].sum() == 0
    assert restored[:, :, 0].sum() == 0
    assert restored[1:5, 1:5, 1:5].sum() == 64


class _DummyModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = images.shape
        logits = torch.zeros(batch_size, 4, depth, height, width, device=images.device)
        logits[:, 3] = 5.0
        return logits


def _small_config() -> dict:
    return {
        "data": {
            "modalities": ["t1", "t1ce", "t2", "flair"],
        },
        "preprocessing": {
            "target_size": [8, 8, 8],
            "normalization": "zscore",
            "crop_to_brain": True,
            "crop_margin": 0,
        },
    }


def _write_modalities(
    root: Path,
    shapes: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, str]:
    shapes = shapes or {}
    paths: dict[str, str] = {}
    for modality in ["t1", "t1ce", "t2", "flair"]:
        shape = shapes.get(modality, (4, 4, 4))
        data = np.zeros(shape, dtype=np.float32)
        data[1:3, 1:3, 1:3] = 10.0
        path = root / f"case_{modality}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        paths[modality] = str(path)
    return paths


def _fresh_tmp_dir() -> Path:
    root = Path("tmp") / f"test_predictor_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
