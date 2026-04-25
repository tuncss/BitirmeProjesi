"""Tests for the 3D Attention U-Net segmentation model."""

from __future__ import annotations

import torch

from src.models import AttentionUNet3D, UNet3D


def test_attention_unet3d_forward_preserves_input_spatial_shape() -> None:
    model = AttentionUNet3D(in_channels=4, num_classes=4, base_filters=2)
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert out.shape == (1, 4, 32, 32, 32)


def test_attention_unet3d_returns_attention_maps() -> None:
    model = AttentionUNet3D(in_channels=4, num_classes=4, base_filters=2)
    x = torch.randn(1, 4, 32, 32, 32)

    attention_maps = model.get_attention_maps(x)

    assert list(attention_maps.keys()) == ["att4", "att3", "att2", "att1"]
    assert attention_maps["att4"].shape == (1, 1, 4, 4, 4)
    assert attention_maps["att3"].shape == (1, 1, 8, 8, 8)
    assert attention_maps["att2"].shape == (1, 1, 16, 16, 16)
    assert attention_maps["att1"].shape == (1, 1, 32, 32, 32)
    assert all(not attention_map.requires_grad for attention_map in attention_maps.values())


def test_attention_unet3d_plan_shape_on_meta_tensor() -> None:
    device = torch.device("meta")
    model = AttentionUNet3D(in_channels=4, num_classes=4, base_filters=32).to(device)
    x = torch.randn(1, 4, 128, 128, 128, device=device)

    out = model(x)
    attention_maps = model.get_attention_maps(x)

    assert out.shape == (1, 4, 128, 128, 128)
    assert attention_maps["att4"].shape == (1, 1, 16, 16, 16)
    assert attention_maps["att3"].shape == (1, 1, 32, 32, 32)
    assert attention_maps["att2"].shape == (1, 1, 64, 64, 64)
    assert attention_maps["att1"].shape == (1, 1, 128, 128, 128)


def test_attention_unet3d_parameter_count_for_default_width() -> None:
    unet_params = sum(parameter.numel() for parameter in UNet3D().parameters())
    attention_params = sum(parameter.numel() for parameter in AttentionUNet3D().parameters())

    assert attention_params == 26_456_416
    assert attention_params > unet_params
