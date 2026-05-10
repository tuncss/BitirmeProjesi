"""Post-processing utilities for BraTS segmentation masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, label


BRATS_OUTPUT_LABELS = (0, 1, 2, 4)
FOREGROUND_LABELS = (1, 2, 4)


class SegmentationPostProcessor:
    """Clean model segmentation masks before saving or serving them.

    The processor expects BraTS output labels: 0=background, 1=NCR/NET,
    2=edema, and 4=enhancing tumor.
    """

    def __init__(
        self,
        min_component_size: int = 100,
        fill_holes: bool = True,
    ):
        if min_component_size < 0:
            raise ValueError(f"min_component_size must be non-negative, got {min_component_size}")

        self.min_component_size = min_component_size
        self.fill_holes = fill_holes

    def process(self, mask: np.ndarray) -> np.ndarray:
        """Apply component cleanup and optional WT hole filling."""
        result = validate_brats_mask(mask).copy()

        if self.min_component_size > 0:
            result = self.remove_small_components_by_class(result)

        if self.fill_holes:
            result = self.fill_whole_tumor_holes(result)

        return validate_brats_mask(result)

    def remove_small_components_by_class(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than the configured threshold."""
        result = validate_brats_mask(mask).copy()

        for class_id in FOREGROUND_LABELS:
            class_mask = result == class_id
            cleaned = self.remove_small_components(class_mask, self.min_component_size)
            removed = class_mask & ~cleaned
            result[removed] = 0

        return result

    @staticmethod
    def remove_small_components(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
        """Return a boolean mask with small connected components removed."""
        if min_size < 0:
            raise ValueError(f"min_size must be non-negative, got {min_size}")

        mask = np.asarray(binary_mask).astype(bool, copy=False)
        if min_size == 0 or not mask.any():
            return mask.copy()

        labeled_array, num_features = label(mask)
        cleaned = np.zeros_like(mask, dtype=bool)
        for component_id in range(1, num_features + 1):
            component = labeled_array == component_id
            if int(component.sum()) >= min_size:
                cleaned[component] = True
        return cleaned

    @staticmethod
    def fill_whole_tumor_holes(mask: np.ndarray) -> np.ndarray:
        """Fill holes inside the WT region and assign new voxels to edema label 2."""
        result = validate_brats_mask(mask).copy()
        whole_tumor = np.isin(result, FOREGROUND_LABELS)
        if not whole_tumor.any():
            return result

        filled = binary_fill_holes(whole_tumor)
        new_voxels = filled & ~whole_tumor
        result[new_voxels] = 2
        return result


def validate_brats_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a segmentation mask uses BraTS output labels and return an array view."""
    array = np.asarray(mask)
    labels = set(np.unique(array).astype(int).tolist())
    unexpected = sorted(labels.difference(BRATS_OUTPUT_LABELS))
    if unexpected:
        raise ValueError(f"Unexpected BraTS output labels: {unexpected}")
    return array
