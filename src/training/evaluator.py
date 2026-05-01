"""Evaluation metrics for BraTS-style 3D segmentation outputs."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


BRATS_REGIONS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute per-class Dice scores from integer prediction and target masks."""
    _validate_torch_metric_inputs(pred, target, num_classes)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    scores: dict[str, float] = {}
    for class_index in range(num_classes):
        pred_class = pred_flat == class_index
        target_class = target_flat == class_index
        tp, fp, fn = _torch_overlap_counts(pred_class, target_class)
        scores[f"class_{class_index}"] = _dice_from_counts(tp, fp, fn)

    return scores


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute per-class IoU scores from integer prediction and target masks."""
    _validate_torch_metric_inputs(pred, target, num_classes)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    scores: dict[str, float] = {}
    for class_index in range(num_classes):
        pred_class = pred_flat == class_index
        target_class = target_flat == class_index
        tp, fp, fn = _torch_overlap_counts(pred_class, target_class)
        scores[f"class_{class_index}"] = _iou_from_counts(tp, fp, fn)

    return scores


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute the 95th percentile symmetric Hausdorff distance in voxels."""
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    if pred_bool.shape != target_bool.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred_bool.shape} and {target_bool.shape}"
        )
    if pred_bool.sum() == 0 or target_bool.sum() == 0:
        return float("inf")

    pred_surface = pred_bool ^ binary_erosion(pred_bool, border_value=0)
    target_surface = target_bool ^ binary_erosion(target_bool, border_value=0)

    pred_dist = distance_transform_edt(~target_bool)
    target_dist = distance_transform_edt(~pred_bool)

    pred_to_target = pred_dist[pred_surface]
    target_to_pred = target_dist[target_surface]
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    if all_distances.size == 0:
        return 0.0

    return float(np.percentile(all_distances, 95))


def compute_brats_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    include_hd95: bool = True,
) -> dict[str, float]:
    """Compute BraTS WT/TC/ET Dice, IoU, and HD95 metrics.

    The masks are expected to use converted training labels:
    0=background, 1=NCR/NET, 2=ED, 3=ET.
    """
    pred_array = np.asarray(pred_mask)
    gt_array = np.asarray(gt_mask)
    if pred_array.shape != gt_array.shape:
        raise ValueError(
            f"pred_mask and gt_mask must have the same shape, got {pred_array.shape} and {gt_array.shape}"
        )

    metrics: dict[str, float] = {}
    dice_values: list[float] = []

    for region_name, labels in BRATS_REGIONS.items():
        pred_binary = np.isin(pred_array, labels)
        gt_binary = np.isin(gt_array, labels)
        tp, fp, fn = _numpy_overlap_counts(pred_binary, gt_binary)

        dice = _dice_from_counts(tp, fp, fn)
        iou = _iou_from_counts(tp, fp, fn)
        metrics[f"{region_name}_dice"] = dice
        metrics[f"{region_name}_iou"] = iou
        if include_hd95:
            metrics[f"{region_name}_hd95"] = hausdorff_distance_95(pred_binary, gt_binary)
        dice_values.append(dice)

    metrics["mean_dice"] = float(np.mean(dice_values))
    return metrics


class MetricTracker:
    """Accumulate metric dictionaries and return arithmetic means."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.metrics_sum: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            value = float(value)
            if math.isnan(value):
                continue
            self.metrics_sum[key] = self.metrics_sum.get(key, 0.0) + value
        self.count += 1

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.metrics_sum.items()}


def _validate_torch_metric_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} and {target.shape}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")


def _torch_overlap_counts(
    pred_class: torch.Tensor,
    target_class: torch.Tensor,
) -> tuple[float, float, float]:
    pred_float = pred_class.float()
    target_float = target_class.float()
    tp = (pred_float * target_float).sum().item()
    fp = (pred_float * (1.0 - target_float)).sum().item()
    fn = ((1.0 - pred_float) * target_float).sum().item()
    return float(tp), float(fp), float(fn)


def _numpy_overlap_counts(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
) -> tuple[float, float, float]:
    pred_bool = np.asarray(pred_binary).astype(bool)
    target_bool = np.asarray(target_binary).astype(bool)
    tp = np.logical_and(pred_bool, target_bool).sum()
    fp = np.logical_and(pred_bool, ~target_bool).sum()
    fn = np.logical_and(~pred_bool, target_bool).sum()
    return float(tp), float(fp), float(fn)


def _dice_from_counts(tp: float, fp: float, fn: float) -> float:
    denominator = 2.0 * tp + fp + fn
    if denominator == 0:
        return 1.0
    return float((2.0 * tp) / denominator)


def _iou_from_counts(tp: float, fp: float, fn: float) -> float:
    denominator = tp + fp + fn
    if denominator == 0:
        return 1.0
    return float(tp / denominator)
