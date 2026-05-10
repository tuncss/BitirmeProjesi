"""Tests for segmentation post-processing."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import SegmentationPostProcessor, validate_brats_mask


def test_remove_small_components_removes_noise_and_keeps_large_component() -> None:
    mask = np.zeros((16, 16, 16), dtype=np.int16)
    mask[4:10, 4:10, 4:10] = 1
    mask[1, 1, 1] = 1

    processor = SegmentationPostProcessor(min_component_size=20, fill_holes=False)
    result = processor.process(mask)

    assert result[1, 1, 1] == 0
    assert result[6, 6, 6] == 1
    assert set(np.unique(result).tolist()) == {0, 1}


def test_remove_small_components_is_applied_per_foreground_class() -> None:
    mask = np.zeros((16, 16, 16), dtype=np.int16)
    mask[2:7, 2:7, 2:7] = 2
    mask[10:14, 10:14, 10:14] = 4
    mask[1, 1, 1] = 2
    mask[15, 15, 15] = 4

    processor = SegmentationPostProcessor(min_component_size=10, fill_holes=False)
    result = processor.process(mask)

    assert result[1, 1, 1] == 0
    assert result[15, 15, 15] == 0
    assert result[3, 3, 3] == 2
    assert result[11, 11, 11] == 4


def test_fill_whole_tumor_holes_assigns_new_voxels_to_edema() -> None:
    mask = np.zeros((9, 9, 9), dtype=np.int16)
    mask[2:7, 2:7, 2:7] = 1
    mask[4, 4, 4] = 0

    processor = SegmentationPostProcessor(min_component_size=0, fill_holes=True)
    result = processor.process(mask)

    assert result[4, 4, 4] == 2
    assert result[3, 3, 3] == 1
    assert set(np.unique(result).tolist()) == {0, 1, 2}


def test_fill_holes_can_be_disabled() -> None:
    mask = np.zeros((9, 9, 9), dtype=np.int16)
    mask[2:7, 2:7, 2:7] = 1
    mask[4, 4, 4] = 0

    processor = SegmentationPostProcessor(min_component_size=0, fill_holes=False)
    result = processor.process(mask)

    assert result[4, 4, 4] == 0


def test_validate_brats_mask_rejects_training_label_three() -> None:
    mask = np.array([0, 1, 2, 3])

    with pytest.raises(ValueError, match="Unexpected BraTS output labels"):
        validate_brats_mask(mask)


def test_processor_rejects_negative_component_size() -> None:
    with pytest.raises(ValueError, match="min_component_size"):
        SegmentationPostProcessor(min_component_size=-1)


def test_remove_small_components_accepts_empty_masks() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.int16)

    processor = SegmentationPostProcessor(min_component_size=10)
    result = processor.process(mask)

    assert np.array_equal(result, mask)
