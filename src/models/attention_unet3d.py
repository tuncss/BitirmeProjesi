"""3D Attention U-Net segmentation model."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from src.models.blocks import AttentionGate, ConvBlock3D, EncoderBlock


class AttentionUNet3D(nn.Module):
    """3D Attention U-Net for multi-class brain tumor segmentation.

    It follows the same encoder and decoder width progression as ``UNet3D``.
    Each decoder skip connection is filtered by an attention gate before
    concatenation, and outputs are raw class logits.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
    ):
        super().__init__()
        if base_filters < 2:
            raise ValueError("base_filters must be at least 2 for attention bottlenecks")

        f = base_filters

        self.encoder1 = EncoderBlock(in_channels, f)
        self.encoder2 = EncoderBlock(f, f * 2)
        self.encoder3 = EncoderBlock(f * 2, f * 4)
        self.encoder4 = EncoderBlock(f * 4, f * 8)

        self.bottleneck = ConvBlock3D(f * 8, f * 16)

        self.up4 = nn.ConvTranspose3d(f * 16, f * 16, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 8, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 4, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(f * 2, f * 2, kernel_size=2, stride=2)

        self.att4 = AttentionGate(F_g=f * 16, F_l=f * 8, F_int=f * 4)
        self.att3 = AttentionGate(F_g=f * 8, F_l=f * 4, F_int=f * 2)
        self.att2 = AttentionGate(F_g=f * 4, F_l=f * 2, F_int=f)
        self.att1 = AttentionGate(F_g=f * 2, F_l=f, F_int=f // 2)

        self.dec_conv4 = ConvBlock3D(f * 16 + f * 8, f * 8)
        self.dec_conv3 = ConvBlock3D(f * 8 + f * 4, f * 4)
        self.dec_conv2 = ConvBlock3D(f * 4 + f * 2, f * 2)
        self.dec_conv1 = ConvBlock3D(f * 2 + f, f)

        self.final_conv = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.bottleneck(x)

        g4 = self.up4(x)
        skip4_att = self.att4(g=x, x=skip4)
        d4 = torch.cat([g4, skip4_att], dim=1)
        d4 = self.dec_conv4(d4)

        g3 = self.up3(d4)
        skip3_att = self.att3(g=d4, x=skip3)
        d3 = torch.cat([g3, skip3_att], dim=1)
        d3 = self.dec_conv3(d3)

        g2 = self.up2(d3)
        skip2_att = self.att2(g=d3, x=skip2)
        d2 = torch.cat([g2, skip2_att], dim=1)
        d2 = self.dec_conv2(d2)

        g1 = self.up1(d2)
        skip1_att = self.att1(g=d2, x=skip1)
        d1 = torch.cat([g1, skip1_att], dim=1)
        d1 = self.dec_conv1(d1)

        return self.final_conv(d1)

    def get_attention_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return attention maps from all decoder levels for visualization."""
        attention_maps: dict[str, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def hook_fn(name: str) -> Callable:
            def fn(
                module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                attention_maps[name] = output.detach()

            return fn

        try:
            hooks.append(self.att4.psi.register_forward_hook(hook_fn("att4")))
            hooks.append(self.att3.psi.register_forward_hook(hook_fn("att3")))
            hooks.append(self.att2.psi.register_forward_hook(hook_fn("att2")))
            hooks.append(self.att1.psi.register_forward_hook(hook_fn("att1")))
            _ = self.forward(x)
        finally:
            for hook in hooks:
                hook.remove()

        return attention_maps
