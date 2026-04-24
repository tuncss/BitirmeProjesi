"""Preprocessing pipeline for BraTS 2021 3D tumor segmentation data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import ndimage
from tqdm import tqdm

try:
    from src.data.utils import load_nifti
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from utils import load_nifti


BRATS_TO_TRAIN_LABELS = {0: 0, 1: 1, 2: 2, 4: 3}
TRAIN_TO_BRATS_LABELS = {0: 0, 1: 1, 2: 2, 3: 4}
DEFAULT_MODALITIES = ("t1", "t1ce", "t2", "flair")


class BraTSPreprocessor:
    """Preprocess BraTS cases into model-ready arrays."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        preprocessing_config = config.get("preprocessing", {})
        data_config = config.get("data", {})

        self.target_size = tuple(preprocessing_config.get("target_size", [128, 128, 128]))
        self.normalization = preprocessing_config.get("normalization", "zscore")
        self.crop_to_brain = bool(preprocessing_config.get("crop_to_brain", True))
        self.modalities = tuple(data_config.get("modalities", DEFAULT_MODALITIES))
        self.crop_margin = int(preprocessing_config.get("crop_margin", 5))

        if self.normalization != "zscore":
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        if len(self.target_size) != 3:
            raise ValueError(f"target_size must contain 3 values, got: {self.target_size}")

    def preprocess_case(self, case_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Process a single BraTS case into images and mask arrays.

        Returns:
            images: float32 array with shape (4, 128, 128, 128)
            mask: int64 array with shape (128, 128, 128), labels {0, 1, 2, 3}
        """
        case_path = Path(case_dir)
        if not case_path.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_path}")

        volumes = [self._load_modality(case_path, modality) for modality in self.modalities]
        mask = self._load_modality(case_path, "seg").astype(np.int16, copy=False)

        self._validate_case_shapes(case_path, volumes, mask)

        if self.crop_to_brain:
            volumes, mask = self.crop_to_nonzero(volumes, mask)

        resized_volumes = [
            self.resize_volume(volume, self.target_size, is_mask=False)
            for volume in volumes
        ]
        resized_mask = self.resize_volume(mask, self.target_size, is_mask=True)

        normalized_volumes = [
            self.zscore_normalize(volume) for volume in resized_volumes
        ]
        images = np.stack(normalized_volumes, axis=0).astype(np.float32, copy=False)
        converted_mask = self.convert_labels(resized_mask).astype(np.int64, copy=False)

        if images.shape != (len(self.modalities), *self.target_size):
            raise ValueError(f"Unexpected image shape for {case_path.name}: {images.shape}")
        if converted_mask.shape != self.target_size:
            raise ValueError(f"Unexpected mask shape for {case_path.name}: {converted_mask.shape}")

        return images, converted_mask

    def zscore_normalize(self, volume: np.ndarray) -> np.ndarray:
        """Apply z-score normalization on non-zero brain voxels only."""
        normalized = np.zeros_like(volume, dtype=np.float32)
        brain = volume != 0
        if not brain.any():
            return normalized

        values = volume[brain].astype(np.float32, copy=False)
        mean = values.mean()
        std = values.std()
        normalized[brain] = (values - mean) / (std + 1e-8)
        return normalized

    def crop_to_nonzero(
        self,
        volumes: list[np.ndarray],
        mask: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Crop all modalities and mask to the shared non-zero brain bounding box."""
        combined = np.zeros_like(volumes[0], dtype=bool)
        for volume in volumes:
            combined |= volume != 0

        if not combined.any():
            return volumes, mask

        coords = np.argwhere(combined)
        mins = np.maximum(coords.min(axis=0) - self.crop_margin, 0)
        maxs = np.minimum(coords.max(axis=0) + self.crop_margin + 1, combined.shape)
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs))

        cropped_volumes = [volume[slices] for volume in volumes]
        cropped_mask = mask[slices]
        return cropped_volumes, cropped_mask

    def resize_volume(
        self,
        volume: np.ndarray,
        target: tuple[int, int, int],
        is_mask: bool = False,
    ) -> np.ndarray:
        """Resize a 3D volume to target shape.

        Cubic interpolation is used for image volumes. Nearest-neighbor
        interpolation is used for masks so segmentation labels stay intact.
        """
        target_array = np.array(target, dtype=np.float32)
        source_array = np.array(volume.shape, dtype=np.float32)
        zoom_factors = target_array / source_array
        order = 0 if is_mask else 3

        resized = ndimage.zoom(volume, zoom=zoom_factors, order=order, mode="constant", cval=0)
        resized = self._fit_to_shape(resized, target)

        if is_mask:
            return np.rint(resized).astype(volume.dtype, copy=False)

        brain_mask = ndimage.zoom(
            (volume != 0).astype(np.uint8),
            zoom=zoom_factors,
            order=0,
            mode="constant",
            cval=0,
        )
        brain_mask = self._fit_to_shape(brain_mask, target).astype(bool, copy=False)
        resized[~brain_mask] = 0
        return resized.astype(np.float32, copy=False)

    def convert_labels(self, mask: np.ndarray) -> np.ndarray:
        """Convert BraTS labels {0, 1, 2, 4} to train labels {0, 1, 2, 3}."""
        converted = np.zeros_like(mask, dtype=np.int64)
        labels = set(np.unique(mask).astype(int).tolist())
        unexpected = labels.difference(BRATS_TO_TRAIN_LABELS)
        if unexpected:
            raise ValueError(f"Unexpected BraTS labels found: {sorted(unexpected)}")

        for source, target in BRATS_TO_TRAIN_LABELS.items():
            converted[mask == source] = target
        return converted

    def inverse_convert_labels(self, mask: np.ndarray) -> np.ndarray:
        """Convert train labels {0, 1, 2, 3} back to BraTS labels {0, 1, 2, 4}."""
        restored = np.zeros_like(mask, dtype=np.int16)
        labels = set(np.unique(mask).astype(int).tolist())
        unexpected = labels.difference(TRAIN_TO_BRATS_LABELS)
        if unexpected:
            raise ValueError(f"Unexpected training labels found: {sorted(unexpected)}")

        for source, target in TRAIN_TO_BRATS_LABELS.items():
            restored[mask == source] = target
        return restored

    def preprocess_dataset(
        self,
        raw_dir: str | Path,
        output_dir: str | Path,
        limit: int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Preprocess all BraTS cases under raw_dir and save compressed .npz files."""
        raw_path = Path(raw_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        case_dirs = sorted(path for path in raw_path.iterdir() if path.is_dir())
        if limit is not None:
            case_dirs = case_dirs[:limit]

        started = time.time()
        succeeded: list[str] = []
        failed: list[dict[str, str]] = []

        for case_dir in tqdm(case_dirs, desc="Preprocessing BraTS cases"):
            output_file = output_path / f"{case_dir.name}.npz"
            if output_file.exists() and not overwrite:
                succeeded.append(case_dir.name)
                continue

            try:
                images, mask = self.preprocess_case(case_dir)
                np.savez_compressed(output_file, images=images, mask=mask)
                succeeded.append(case_dir.name)
            except Exception as exc:  # noqa: BLE001 - keep processing remaining cases
                failed.append({"case_id": case_dir.name, "error": str(exc)})

        elapsed_seconds = time.time() - started
        summary = {
            "total_cases": len(case_dirs),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "elapsed_seconds": elapsed_seconds,
            "failures": failed,
        }

        print(
            "Preprocessing complete: "
            f"{summary['succeeded']}/{summary['total_cases']} succeeded, "
            f"{summary['failed']} failed, "
            f"{elapsed_seconds:.1f}s elapsed"
        )
        if failed:
            print("First failures:")
            for failure in failed[:10]:
                print(f"  - {failure['case_id']}: {failure['error']}")

        return summary

    def _load_modality(self, case_path: Path, modality: str) -> np.ndarray:
        path = case_path / f"{case_path.name}_{modality}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        data, _ = load_nifti(path)
        return data

    @staticmethod
    def _validate_case_shapes(case_path: Path, volumes: list[np.ndarray], mask: np.ndarray) -> None:
        shapes = [volume.shape for volume in volumes] + [mask.shape]
        if len(set(shapes)) != 1:
            raise ValueError(f"Shape mismatch in {case_path.name}: {shapes}")

    @staticmethod
    def _fit_to_shape(volume: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
        """Center-crop or zero-pad a volume when interpolation rounds by one voxel."""
        result = volume
        for axis, target_size in enumerate(target):
            current_size = result.shape[axis]
            if current_size == target_size:
                continue

            if current_size > target_size:
                start = (current_size - target_size) // 2
                stop = start + target_size
                slices = [slice(None)] * result.ndim
                slices[axis] = slice(start, stop)
                result = result[tuple(slices)]
            else:
                before = (target_size - current_size) // 2
                after = target_size - current_size - before
                pad_width = [(0, 0)] * result.ndim
                pad_width[axis] = (before, after)
                result = np.pad(result, pad_width, mode="constant", constant_values=0)
        return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BraTS 2021 cases.")
    parser.add_argument("--config", default="configs/training_config.yaml", type=Path)
    parser.add_argument("--raw-dir", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--case-id", default=None, help="Process a single case by id.")
    parser.add_argument("--limit", default=None, type=int, help="Limit number of cases.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    raw_dir = args.raw_dir or Path(config["data"]["data_dir"])
    output_dir = args.output_dir or Path(config["data"]["processed_dir"])
    preprocessor = BraTSPreprocessor(config)

    if args.case_id:
        case_dir = Path(raw_dir) / args.case_id
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        images, mask = preprocessor.preprocess_case(case_dir)
        np.savez_compressed(output_path / f"{args.case_id}.npz", images=images, mask=mask)
        print(
            f"Saved {args.case_id}: images={images.shape} {images.dtype}, "
            f"mask={mask.shape} {mask.dtype}, labels={np.unique(mask).tolist()}"
        )
        return 0

    summary = preprocessor.preprocess_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
