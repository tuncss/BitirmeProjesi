"""Tests for segmentation evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.training import (
    MetricTracker,
    compute_brats_metrics,
    dice_score,
    hausdorff_distance_95,
    iou_score,
)


def test_dice_score_perfect_match_is_one() -> None:
    target = torch.tensor([[[0, 1], [2, 3]]])
    pred = target.clone()

    scores = dice_score(pred, target, num_classes=4)

    assert scores == {
        "class_0": 1.0,
        "class_1": 1.0,
        "class_2": 1.0,
        "class_3": 1.0,
    }


def test_dice_score_completely_wrong_foreground_is_zero() -> None:
    target = torch.ones(1, 2, 2, dtype=torch.long)
    pred = torch.full_like(target, 2)

    scores = dice_score(pred, target, num_classes=3)

    assert scores["class_1"] == 0.0
    assert scores["class_2"] == 0.0


def test_iou_score_matches_known_counts() -> None:
    target = torch.tensor([[0, 1, 1, 2]])
    pred = torch.tensor([[0, 1, 2, 2]])

    scores = iou_score(pred, target, num_classes=3)

    assert scores["class_0"] == 1.0
    assert scores["class_1"] == pytest.approx(0.5)
    assert scores["class_2"] == pytest.approx(0.5)


def test_compute_brats_metrics_perfect_match() -> None:
    gt = np.array([0, 0, 1, 2, 3, 1, 0, 0, 3, 2])
    pred = gt.copy()

    metrics = compute_brats_metrics(pred, gt)

    assert metrics["WT_dice"] == 1.0
    assert metrics["TC_dice"] == 1.0
    assert metrics["ET_dice"] == 1.0
    assert metrics["WT_iou"] == 1.0
    assert metrics["TC_iou"] == 1.0
    assert metrics["ET_iou"] == 1.0
    assert metrics["mean_dice"] == 1.0


def test_compute_brats_metrics_uses_converted_brats_regions() -> None:
    gt = np.array([0, 1, 2, 3, 0, 0])
    pred = np.array([0, 1, 0, 3, 2, 0])

    metrics = compute_brats_metrics(pred, gt)

    assert metrics["WT_dice"] == pytest.approx(4 / 6)
    assert metrics["WT_iou"] == pytest.approx(2 / 4)
    assert metrics["TC_dice"] == 1.0
    assert metrics["TC_iou"] == 1.0
    assert metrics["ET_dice"] == 1.0
    assert metrics["ET_iou"] == 1.0
    assert metrics["mean_dice"] == pytest.approx(((4 / 6) + 1.0 + 1.0) / 3)


def test_compute_brats_metrics_can_skip_hd95() -> None:
    gt = np.array([0, 1, 2, 3])
    pred = gt.copy()

    metrics = compute_brats_metrics(pred, gt, include_hd95=False)

    assert metrics["WT_dice"] == 1.0
    assert metrics["TC_iou"] == 1.0
    assert "WT_hd95" not in metrics
    assert "TC_hd95" not in metrics
    assert "ET_hd95" not in metrics


def test_hausdorff_distance_95_handles_empty_masks() -> None:
    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    target = np.zeros((4, 4, 4), dtype=np.uint8)
    target[1, 1, 1] = 1

    assert math.isinf(hausdorff_distance_95(pred, target))
    assert math.isinf(hausdorff_distance_95(target, pred))


def test_hausdorff_distance_95_is_zero_for_identical_masks() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1

    assert hausdorff_distance_95(mask, mask) == 0.0


def test_metric_tracker_computes_arithmetic_means() -> None:
    tracker = MetricTracker()

    tracker.update({"WT_dice": 0.90, "TC_dice": 0.85})
    tracker.update({"WT_dice": 0.92, "TC_dice": 0.87})

    assert tracker.compute() == {"WT_dice": 0.91, "TC_dice": 0.86}


def test_metric_tracker_reset_clears_state() -> None:
    tracker = MetricTracker()
    tracker.update({"WT_dice": 0.90})

    tracker.reset()

    assert tracker.compute() == {}


def test_metric_functions_validate_shapes() -> None:
    pred_torch = torch.zeros(1, 2, 2)
    target_torch = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="same shape"):
        dice_score(pred_torch, target_torch, num_classes=2)

    pred_np = np.zeros((2, 2))
    target_np = np.zeros((2, 3))
    with pytest.raises(ValueError, match="same shape"):
        compute_brats_metrics(pred_np, target_np)
