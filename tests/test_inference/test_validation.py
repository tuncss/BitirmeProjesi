"""Tests for BraTS validation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.inference import binarize_region, compute_validation_metrics, dice_score, hausdorff95


def test_binarize_region_uses_brats_region_definitions() -> None:
    mask = np.array([0, 1, 2, 4])

    assert binarize_region(mask, "WT").tolist() == [False, True, True, True]
    assert binarize_region(mask, "TC").tolist() == [False, True, False, True]
    assert binarize_region(mask, "ET").tolist() == [False, False, False, True]


def test_binarize_region_rejects_unknown_region() -> None:
    with pytest.raises(ValueError, match="Unknown BraTS validation region"):
        binarize_region(np.zeros((2, 2, 2)), "NCR")


def test_dice_score_matches_manual_overlap() -> None:
    pred = np.array([True, True, True, False])
    gt = np.array([True, False, True, True])

    assert dice_score(pred, gt) == pytest.approx(4 / 6)


def test_dice_score_handles_empty_masks() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)
    one_voxel = empty.copy()
    one_voxel[1, 1, 1] = True

    assert dice_score(empty, empty) == 1.0
    assert dice_score(one_voxel, empty) == 0.0


def test_hausdorff95_is_zero_for_identical_masks() -> None:
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2:4, 2:4, 2:4] = True

    assert hausdorff95(mask, mask) == pytest.approx(0.0)


def test_hausdorff95_matches_one_voxel_translation() -> None:
    pred = np.zeros((5, 5, 5), dtype=bool)
    gt = np.zeros_like(pred)
    pred[1, 1, 1] = True
    gt[2, 1, 1] = True

    assert hausdorff95(pred, gt) == pytest.approx(1.0)


def test_hausdorff95_uses_spacing() -> None:
    pred = np.zeros((5, 5, 5), dtype=bool)
    gt = np.zeros_like(pred)
    pred[1, 1, 1] = True
    gt[2, 1, 1] = True

    assert hausdorff95(pred, gt, spacing=(2.0, 1.0, 1.0)) == pytest.approx(2.0)


def test_hausdorff95_handles_empty_masks() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)
    one_voxel = empty.copy()
    one_voxel[1, 1, 1] = True

    assert hausdorff95(empty, empty) == 0.0
    assert math.isinf(hausdorff95(one_voxel, empty))


def test_metrics_raise_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        dice_score(np.zeros((2, 2)), np.zeros((2, 3)))

    with pytest.raises(ValueError, match="shapes must match"):
        hausdorff95(np.zeros((2, 2)), np.zeros((2, 3)))

    with pytest.raises(ValueError, match="shapes must match"):
        compute_validation_metrics(np.zeros((2, 2, 2)), np.zeros((3, 2, 2)))


def test_compute_validation_metrics_returns_expected_regions_and_sentinel_hd95() -> None:
    pred = np.zeros((4, 4, 4), dtype=np.int16)
    gt = np.zeros_like(pred)
    pred[1, 1, 1] = 1
    pred[2, 2, 2] = 2
    gt[1, 1, 1] = 1
    gt[3, 3, 3] = 4

    metrics = compute_validation_metrics(pred, gt)

    assert set(metrics) == {"WT", "TC", "ET"}
    assert metrics["WT"]["dice"] == pytest.approx(0.5)
    assert metrics["TC"]["dice"] == pytest.approx(2 / 3)
    assert metrics["ET"]["dice"] == 0.0
    assert metrics["ET"]["hd95"] == -1.0
