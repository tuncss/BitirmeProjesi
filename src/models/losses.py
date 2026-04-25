"""Loss functions for 3D multi-class tumor segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation logits."""

    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss from raw logits and integer label targets."""
        probs, targets_one_hot = _prepare_probabilities_and_targets(predictions, targets)
        probs, targets_one_hot = _drop_background_if_needed(
            probs,
            targets_one_hot,
            include_background=self.include_background,
        )

        reduce_dims = tuple(range(2, predictions.ndim))
        intersection = (probs * targets_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + targets_one_hot.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Weighted sum of soft Dice loss and CrossEntropy loss."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = False,
        ce_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(
            smooth=smooth,
            include_background=include_background,
        )
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce


class TverskyLoss(nn.Module):
    """Tversky loss with configurable false-positive and false-negative weights."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets_one_hot = _prepare_probabilities_and_targets(predictions, targets)
        probs, targets_one_hot = _drop_background_if_needed(
            probs,
            targets_one_hot,
            include_background=False,
        )

        reduce_dims = tuple(range(2, predictions.ndim))
        true_positive = (probs * targets_one_hot).sum(dim=reduce_dims)
        false_positive = (probs * (1.0 - targets_one_hot)).sum(dim=reduce_dims)
        false_negative = ((1.0 - probs) * targets_one_hot).sum(dim=reduce_dims)

        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return 1.0 - tversky.mean()


def _prepare_probabilities_and_targets(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predictions.ndim < 3:
        raise ValueError(
            "predictions must have shape (B, C, ...), "
            f"got {tuple(predictions.shape)}"
        )
    if targets.shape != (predictions.shape[0], *predictions.shape[2:]):
        raise ValueError(
            "targets must have shape (B, ...), matching prediction spatial dimensions; "
            f"got targets={tuple(targets.shape)}, predictions={tuple(predictions.shape)}"
        )

    num_classes = predictions.shape[1]
    probs = F.softmax(predictions, dim=1)
    targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
    targets_one_hot = targets_one_hot.movedim(-1, 1).to(dtype=probs.dtype)
    return probs, targets_one_hot


def _drop_background_if_needed(
    probs: torch.Tensor,
    targets_one_hot: torch.Tensor,
    include_background: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if include_background:
        return probs, targets_one_hot
    if probs.shape[1] <= 1:
        raise ValueError("include_background=False requires at least two classes")
    return probs[:, 1:], targets_one_hot[:, 1:]
