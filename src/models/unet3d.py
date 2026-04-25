"""3D U-Net segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import ConvBlock3D, DecoderBlock, EncoderBlock


class UNet3D(nn.Module):
    """Baseline 3D U-Net for multi-class brain tumor segmentation.

    The model accepts four BraTS modalities and returns raw class logits.
    Spatial dimensions are preserved by same-padded convolutions and four
    encoder/decoder levels.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
    ):
        super().__init__()

        f = base_filters

        self.encoder1 = EncoderBlock(in_channels, f)
        self.encoder2 = EncoderBlock(f, f * 2)
        self.encoder3 = EncoderBlock(f * 2, f * 4)
        self.encoder4 = EncoderBlock(f * 4, f * 8)

        self.bottleneck = ConvBlock3D(f * 8, f * 16)

        self.decoder4 = DecoderBlock(f * 16, f * 8, f * 8)
        self.decoder3 = DecoderBlock(f * 8, f * 4, f * 4)
        self.decoder2 = DecoderBlock(f * 4, f * 2, f * 2)
        self.decoder1 = DecoderBlock(f * 2, f, f)

        self.final_conv = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.final_conv(x)
