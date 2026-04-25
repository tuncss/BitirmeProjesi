"""Reusable 3D building blocks for U-Net style segmentation models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two 3D convolutions with InstanceNorm and LeakyReLU activations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block that returns pooled features and the skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv_block(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution, skip concat, and ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv_block = ConvBlock3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _match_spatial_shape(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class AttentionGate(nn.Module):
    """Attention gate that filters encoder skip features using a decoder signal."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1,
                size=x1.shape[2:],
                mode="trilinear",
                align_corners=True,
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


def _match_spatial_shape(
    x: torch.Tensor,
    target_shape: torch.Size | tuple[int, int, int],
) -> torch.Tensor:
    """Center pad or crop a 5D tensor so its spatial shape matches target_shape."""
    diff_d = int(target_shape[0]) - x.shape[2]
    diff_h = int(target_shape[1]) - x.shape[3]
    diff_w = int(target_shape[2]) - x.shape[4]

    if diff_d == 0 and diff_h == 0 and diff_w == 0:
        return x

    pad = [
        max(diff_w // 2, 0),
        max(diff_w - diff_w // 2, 0),
        max(diff_h // 2, 0),
        max(diff_h - diff_h // 2, 0),
        max(diff_d // 2, 0),
        max(diff_d - diff_d // 2, 0),
    ]
    if any(pad):
        x = F.pad(x, pad)

    crop_d = x.shape[2] - int(target_shape[0])
    crop_h = x.shape[3] - int(target_shape[1])
    crop_w = x.shape[4] - int(target_shape[2])
    start_d = max(crop_d // 2, 0)
    start_h = max(crop_h // 2, 0)
    start_w = max(crop_w // 2, 0)
    end_d = start_d + int(target_shape[0])
    end_h = start_h + int(target_shape[1])
    end_w = start_w + int(target_shape[2])
    return x[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
