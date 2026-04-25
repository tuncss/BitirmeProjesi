"""Tests for reusable 3D model blocks."""

from __future__ import annotations

import torch

from src.models.blocks import AttentionGate, ConvBlock3D, DecoderBlock, EncoderBlock


def test_conv_block_3d_preserves_spatial_shape() -> None:
    x = torch.randn(1, 4, 16, 16, 16)
    block = ConvBlock3D(4, 8)

    out = block(x)

    assert out.shape == (1, 8, 16, 16, 16)


def test_encoder_block_returns_pooled_and_skip_features() -> None:
    x = torch.randn(1, 4, 16, 16, 16)
    block = EncoderBlock(4, 8)

    pooled, features = block(x)

    assert pooled.shape == (1, 8, 8, 8, 8)
    assert features.shape == (1, 8, 16, 16, 16)


def test_decoder_block_upsamples_and_concatenates_skip_features() -> None:
    x = torch.randn(1, 16, 8, 8, 8)
    skip = torch.randn(1, 8, 16, 16, 16)
    block = DecoderBlock(16, 8, 8)

    out = block(x, skip)

    assert out.shape == (1, 8, 16, 16, 16)


def test_decoder_block_handles_odd_skip_shape() -> None:
    x = torch.randn(1, 16, 8, 8, 8)
    skip = torch.randn(1, 8, 17, 15, 16)
    block = DecoderBlock(16, 8, 8)

    out = block(x, skip)

    assert out.shape == (1, 8, 17, 15, 16)


def test_attention_gate_filters_skip_features() -> None:
    g = torch.randn(1, 16, 8, 8, 8)
    x = torch.randn(1, 8, 16, 16, 16)
    gate = AttentionGate(F_g=16, F_l=8, F_int=4)

    out = gate(g, x)

    assert out.shape == x.shape


def test_plan_shapes_on_meta_tensors() -> None:
    device = torch.device("meta")
    x = torch.randn(1, 32, 64, 64, 64, device=device)

    conv = ConvBlock3D(32, 64).to(device)
    assert conv(x).shape == (1, 64, 64, 64, 64)

    encoder = EncoderBlock(32, 64).to(device)
    pooled, features = encoder(x)
    assert pooled.shape == (1, 64, 32, 32, 32)
    assert features.shape == (1, 64, 64, 64, 64)

    decoder = DecoderBlock(128, 64, 64).to(device)
    dec_in = torch.randn(1, 128, 32, 32, 32, device=device)
    assert decoder(dec_in, features).shape == (1, 64, 64, 64, 64)

    gate = AttentionGate(F_g=128, F_l=64, F_int=32).to(device)
    g = torch.randn(1, 128, 32, 32, 32, device=device)
    assert gate(g, features).shape == (1, 64, 64, 64, 64)
