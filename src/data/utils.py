"""Utility functions for BraTS NIfTI data handling."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return data plus affine matrix."""
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return data.astype(np.float32, copy=False), image.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array as a NIfTI file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    nib.save(image, str(output_path))


def get_brain_mask(volume: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Create a binary brain mask from non-background voxels."""
    mask = volume > threshold
    if mask.any():
        mask = ndimage.binary_fill_holes(mask)
    return mask.astype(bool, copy=False)


def compute_voxel_volume(affine: np.ndarray) -> float:
    """Compute voxel volume in mm^3 from an affine matrix."""
    return float(abs(np.linalg.det(affine[:3, :3])))

