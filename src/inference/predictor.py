"""Inference engine for BraTS-style brain tumor segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from src.data.preprocessing import BraTSPreprocessor, DEFAULT_MODALITIES
from src.data.utils import compute_voxel_volume, load_nifti
from src.models import AttentionUNet3D, UNet3D


MODEL_NAMES = ("unet3d", "attention_unet3d")


@dataclass(frozen=True)
class CropBoundingBox:
    """Bounding box used to crop the brain region before model inference."""

    starts: tuple[int, int, int]
    stops: tuple[int, int, int]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class InferenceMetadata:
    """Metadata needed to restore model predictions to the source image space."""

    affine: np.ndarray
    original_shape: tuple[int, int, int]
    crop_bbox: CropBoundingBox
    voxel_volume_mm3: float


class BrainTumorPredictor:
    """Load a trained model and run full-volume tumor segmentation inference."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.preprocessor = BraTSPreprocessor(config or _default_inference_config())
        self.modalities = tuple(self.preprocessor.modalities)
        self.model = self._load_model(model_name, self.checkpoint_path)
        self.model.eval()

        logger.info(f"Predictor initialized: {model_name} on {self.device}")

    @torch.no_grad()
    def predict(self, modality_paths: dict[str, str | Path]) -> dict[str, Any]:
        """Run preprocessing, model inference, restore, and volume measurement."""
        images, metadata = self.preprocess_modalities(modality_paths)
        input_tensor = torch.from_numpy(images).unsqueeze(0).to(self.device)

        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)

        pred_labels = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64, copy=False)
        probabilities_np = probabilities.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        restored_train_labels = self.restore_to_original(pred_labels, metadata)
        brats_mask = self.preprocessor.inverse_convert_labels(restored_train_labels)
        tumor_volumes = compute_tumor_volumes(brats_mask, metadata.voxel_volume_mm3)

        return {
            "segmentation_mask": brats_mask,
            "probabilities": probabilities_np,
            "affine": metadata.affine,
            "original_shape": metadata.original_shape,
            "crop_bbox": metadata.crop_bbox,
            "tumor_volumes": tumor_volumes,
            "model_name": self.model_name,
        }

    def preprocess_modalities(
        self,
        modality_paths: dict[str, str | Path],
    ) -> tuple[np.ndarray, InferenceMetadata]:
        """Load four NIfTI modalities and convert them to a model input tensor."""
        paths = _validate_modality_paths(modality_paths, self.modalities)
        volumes: list[np.ndarray] = []
        affines: list[np.ndarray] = []

        for modality in self.modalities:
            volume, affine = load_nifti(paths[modality])
            volumes.append(volume.astype(np.float32, copy=False))
            affines.append(affine)

        _validate_volume_shapes(volumes)
        affine = affines[0]
        original_shape = tuple(int(size) for size in volumes[0].shape)
        crop_bbox = compute_nonzero_bbox(
            volumes,
            margin=self.preprocessor.crop_margin,
            enabled=self.preprocessor.crop_to_brain,
        )
        cropped = [
            volume[
                crop_bbox.starts[0] : crop_bbox.stops[0],
                crop_bbox.starts[1] : crop_bbox.stops[1],
                crop_bbox.starts[2] : crop_bbox.stops[2],
            ]
            for volume in volumes
        ]

        resized = [
            self.preprocessor.resize_volume(volume, self.preprocessor.target_size, is_mask=False)
            for volume in cropped
        ]
        normalized = [self.preprocessor.zscore_normalize(volume) for volume in resized]
        images = np.stack(normalized, axis=0).astype(np.float32, copy=False)

        expected_shape = (len(self.modalities), *self.preprocessor.target_size)
        if images.shape != expected_shape:
            raise ValueError(f"Unexpected inference input shape: {images.shape}, expected {expected_shape}")

        metadata = InferenceMetadata(
            affine=affine,
            original_shape=original_shape,
            crop_bbox=crop_bbox,
            voxel_volume_mm3=compute_voxel_volume(affine),
        )
        return images, metadata

    def restore_to_original(
        self,
        prediction: np.ndarray,
        metadata: InferenceMetadata,
    ) -> np.ndarray:
        """Restore a target-size prediction back into the original image shape."""
        crop_shape = metadata.crop_bbox.shape
        resized = self.preprocessor.resize_volume(
            prediction.astype(np.int16, copy=False),
            crop_shape,
            is_mask=True,
        ).astype(np.int64, copy=False)

        restored = np.zeros(metadata.original_shape, dtype=np.int64)
        starts = metadata.crop_bbox.starts
        stops = metadata.crop_bbox.stops
        restored[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ] = resized
        return restored

    def _load_model(self, model_name: str, checkpoint_path: Path) -> torch.nn.Module:
        if model_name == "unet3d":
            model = UNet3D(in_channels=len(self.modalities), num_classes=4)
        elif model_name == "attention_unet3d":
            model = AttentionUNet3D(in_channels=len(self.modalities), num_classes=4)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        logger.info(f"Model loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
        return model


def compute_nonzero_bbox(
    volumes: list[np.ndarray],
    margin: int,
    enabled: bool = True,
) -> CropBoundingBox:
    """Compute a shared non-zero bounding box for all modalities."""
    shape = tuple(int(size) for size in volumes[0].shape)
    if not enabled:
        return CropBoundingBox(starts=(0, 0, 0), stops=shape, shape=shape)

    combined = np.zeros(shape, dtype=bool)
    for volume in volumes:
        combined |= volume != 0

    if not combined.any():
        return CropBoundingBox(starts=(0, 0, 0), stops=shape, shape=shape)

    coords = np.argwhere(combined)
    starts_array = np.maximum(coords.min(axis=0) - margin, 0)
    stops_array = np.minimum(coords.max(axis=0) + margin + 1, np.array(shape))
    starts = tuple(int(value) for value in starts_array)
    stops = tuple(int(value) for value in stops_array)
    crop_shape = tuple(stop - start for start, stop in zip(starts, stops))
    return CropBoundingBox(starts=starts, stops=stops, shape=crop_shape)


def compute_tumor_volumes(mask: np.ndarray, voxel_volume_mm3: float) -> dict[str, float]:
    """Compute WT/TC/ET region volumes in cubic centimeters from BraTS labels."""
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0
    wt_voxels = int(np.isin(mask, [1, 2, 4]).sum())
    tc_voxels = int(np.isin(mask, [1, 4]).sum())
    et_voxels = int((mask == 4).sum())
    return {
        "WT_cm3": round(float(wt_voxels * voxel_volume_cm3), 2),
        "TC_cm3": round(float(tc_voxels * voxel_volume_cm3), 2),
        "ET_cm3": round(float(et_voxels * voxel_volume_cm3), 2),
    }


def _default_inference_config() -> dict[str, Any]:
    return {
        "data": {
            "modalities": list(DEFAULT_MODALITIES),
        },
        "preprocessing": {
            "target_size": [128, 128, 128],
            "normalization": "zscore",
            "crop_to_brain": True,
            "crop_margin": 5,
        },
    }


def _validate_modality_paths(
    modality_paths: dict[str, str | Path],
    required_modalities: tuple[str, ...],
) -> dict[str, Path]:
    provided = set(modality_paths)
    required = set(required_modalities)
    missing = sorted(required - provided)
    extra = sorted(provided - required)
    if missing:
        raise ValueError(f"Missing modalities: {missing}")
    if extra:
        raise ValueError(f"Unexpected modalities: {extra}")

    paths = {modality: Path(modality_paths[modality]) for modality in required_modalities}
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing modality files: {missing_files}")
    return paths


def _validate_volume_shapes(volumes: list[np.ndarray]) -> None:
    shapes = [tuple(volume.shape) for volume in volumes]
    if len(set(shapes)) != 1:
        raise ValueError(f"All modalities must have the same shape, got: {shapes}")
