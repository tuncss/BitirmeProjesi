"""Tests for the 3D U-Net segmentation model."""

from __future__ import annotations

import torch

from src.models import UNet3D


def test_unet3d_forward_preserves_input_spatial_shape() -> None:
    model = UNet3D(in_channels=4, num_classes=4, base_filters=2)
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert out.shape == (1, 4, 32, 32, 32)


def test_unet3d_plan_shape_on_meta_tensor() -> None:
    device = torch.device("meta")
    model = UNet3D(in_channels=4, num_classes=4, base_filters=32).to(device)
    x = torch.randn(1, 4, 128, 128, 128, device=device)

    out = model(x)

    assert out.shape == (1, 4, 128, 128, 128)


def test_unet3d_parameter_count_for_default_width() -> None:
    model = UNet3D(in_channels=4, num_classes=4, base_filters=32)

    params = sum(parameter.numel() for parameter in model.parameters())

    assert params == 26_324_164
