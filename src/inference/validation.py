"""Validation metrics for BraTS segmentation masks."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


RegionName = Literal["WT", "TC", "ET"]
VALIDATION_REGIONS: tuple[RegionName, ...] = ("WT", "TC", "ET")


def binarize_region(mask: np.ndarray, region: str) -> np.ndarray:
    """Return a boolean BraTS region mask for WT, TC, or ET."""
    array = np.asarray(mask)
    normalized_region = region.upper()

    if normalized_region == "WT":
        return np.isin(array, (1, 2, 4))
    if normalized_region == "TC":
        return np.isin(array, (1, 4))
    if normalized_region == "ET":
        return array == 4

    raise ValueError(f"Unknown BraTS validation region: {region}")


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Compute the Sorensen-Dice score for two binary masks."""
    pred, gt = _as_matching_binary_masks(pred_bin, gt_bin)
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int(np.logical_and(pred, gt).sum())
    return float((2.0 * intersection) / (pred_sum + gt_sum))


def hausdorff95(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute the symmetric 95th percentile Hausdorff distance.

    Returns 0.0 when both masks are empty and infinity when only one mask is
    empty. Callers that need JSON-safe values should convert infinity to a
    sentinel at the API boundary.
    """
    pred, gt = _as_matching_binary_masks(pred_bin, gt_bin)
    spacing = _normalize_spacing(spacing, pred.ndim)
    pred_empty = not bool(pred.any())
    gt_empty = not bool(gt.any())

    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return float("inf")

    medpy_hd95 = _medpy_hd95()
    if medpy_hd95 is not None:
        return float(medpy_hd95(pred, gt, voxelspacing=spacing))

    pred_surface = _surface_voxels(pred)
    gt_surface = _surface_voxels(gt)

    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((dt_to_gt[pred_surface], dt_to_pred[gt_surface]))
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def compute_validation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, dict[str, float]]:
    """Compute Dice and JSON-safe HD95 metrics for WT, TC, and ET regions."""
    pred_array = np.asarray(pred)
    gt_array = np.asarray(gt)
    if pred_array.shape != gt_array.shape:
        raise ValueError(
            "Prediction and ground-truth shapes must match, "
            f"got {pred_array.shape} and {gt_array.shape}"
        )

    metrics: dict[str, dict[str, float]] = {}
    for region in VALIDATION_REGIONS:
        pred_bin = binarize_region(pred_array, region)
        gt_bin = binarize_region(gt_array, region)
        hd95 = hausdorff95(pred_bin, gt_bin, spacing=spacing)
        metrics[region] = {
            "dice": float(dice_score(pred_bin, gt_bin)),
            "hd95": float(hd95) if math.isfinite(hd95) else -1.0,
        }
    return metrics


def _as_matching_binary_masks(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_array = np.asarray(first).astype(bool, copy=False)
    second_array = np.asarray(second).astype(bool, copy=False)
    if first_array.shape != second_array.shape:
        raise ValueError(
            "Binary mask shapes must match, "
            f"got {first_array.shape} and {second_array.shape}"
        )
    return first_array, second_array


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def _normalize_spacing(spacing: tuple[float, ...], ndim: int) -> tuple[float, ...]:
    if len(spacing) != ndim:
        raise ValueError(f"Spacing must have {ndim} values, got {len(spacing)}")

    normalized = tuple(float(value) for value in spacing)
    if any(value <= 0 for value in normalized):
        raise ValueError(f"Spacing values must be positive, got {spacing}")
    return normalized


def _medpy_hd95():
    try:
        from medpy.metric.binary import hd95
    except ImportError:
        return None
    return hd95
