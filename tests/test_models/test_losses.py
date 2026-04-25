"""Tests for segmentation loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.models import DiceCELoss, DiceLoss, TverskyLoss


def test_dice_loss_backward_produces_gradients() -> None:
    predictions = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    loss_fn = DiceLoss()

    loss = loss_fn(predictions, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_dice_loss_excludes_background_by_default() -> None:
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    targets[:, 0, 0, 0] = 1
    predictions = torch.full((1, 4, 4, 4, 4), -20.0)
    predictions[:, 0] = 20.0

    loss_without_background = DiceLoss(include_background=False)(predictions, targets)
    loss_with_background = DiceLoss(include_background=True)(predictions, targets)

    assert loss_without_background > 0.30
    assert loss_with_background < loss_without_background


def test_dice_loss_perfect_match_is_near_zero() -> None:
    targets = torch.arange(64).reshape(1, 4, 4, 4) % 4
    predictions = torch.full((1, 4, 4, 4, 4), -20.0)
    predictions.scatter_(1, targets.unsqueeze(1), 20.0)

    loss = DiceLoss()(predictions, targets)

    assert loss.item() < 1e-6


def test_dice_loss_completely_wrong_prediction_is_near_one() -> None:
    targets = torch.arange(64).reshape(1, 4, 4, 4) % 4
    wrong_targets = (targets % 3) + 1
    predictions = torch.full((1, 4, 4, 4, 4), -20.0)
    predictions.scatter_(1, wrong_targets.unsqueeze(1), 20.0)

    loss = DiceLoss()(predictions, targets)

    assert loss.item() > 0.99


def test_dice_ce_loss_matches_weighted_components() -> None:
    predictions = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    loss_fn = DiceCELoss(dice_weight=0.7, ce_weight=0.3)

    loss = loss_fn(predictions, targets)
    expected = 0.7 * DiceLoss()(predictions, targets) + 0.3 * F.cross_entropy(
        predictions,
        targets,
    )

    assert torch.allclose(loss, expected)


def test_dice_ce_loss_backward_produces_gradients() -> None:
    predictions = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    loss_fn = DiceCELoss()

    loss = loss_fn(predictions, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_tversky_loss_backward_produces_gradients() -> None:
    predictions = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)

    loss = loss_fn(predictions, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_loss_functions_validate_target_shape() -> None:
    predictions = torch.randn(1, 4, 8, 8, 8)
    targets = torch.randint(0, 4, (1, 1, 8, 8, 8))

    with pytest.raises(ValueError, match="targets must have shape"):
        DiceLoss()(predictions, targets)
