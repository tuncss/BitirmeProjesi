"""Tests for the training CLI helpers."""

from __future__ import annotations

import shutil
import uuid
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from scripts.train import (
    apply_cli_overrides,
    create_loss_fn,
    create_model,
    load_checkpoint,
    parse_args,
    set_seed,
)
from src.models import AttentionUNet3D, DiceCELoss, DiceLoss, TverskyLoss, UNet3D
from src.training import ExperimentConfig


def test_parse_args_supports_training_options() -> None:
    args = parse_args(
        [
            "--model",
            "attention_unet3d",
            "--config",
            "configs/custom.yaml",
            "--resume",
            "data/models/model.pth",
            "--batch-size",
            "4",
            "--epochs",
            "10",
            "--lr",
            "0.001",
            "--seed",
            "123",
            "--experiment-name",
            "trial",
        ]
    )

    assert args.model == "attention_unet3d"
    assert args.config == "configs/custom.yaml"
    assert args.resume == "data/models/model.pth"
    assert args.batch_size == 4
    assert args.epochs == 10
    assert args.lr == 0.001
    assert args.seed == 123
    assert args.experiment_name == "trial"


def test_apply_cli_overrides_updates_config() -> None:
    config = ExperimentConfig()
    args = Namespace(
        model="unet3d",
        batch_size=3,
        epochs=5,
        lr=2e-4,
        seed=99,
        experiment_name="override_test",
    )

    updated = apply_cli_overrides(config, args)

    assert updated.model_name == "unet3d"
    assert updated.training.batch_size == 3
    assert updated.training.epochs == 5
    assert updated.training.learning_rate == 2e-4
    assert updated.seed == 99
    assert updated.experiment_name == "override_test"


def test_create_model_uses_configured_channels_and_classes() -> None:
    config = ExperimentConfig()
    config.data.modalities = ["t1", "t1ce"]
    config.data.num_classes = 3

    unet = create_model("unet3d", config)
    attention_unet = create_model("attention_unet3d", config)

    assert isinstance(unet, UNet3D)
    assert isinstance(attention_unet, AttentionUNet3D)
    assert unet.encoder1.conv_block.conv[0].in_channels == 2
    assert unet.final_conv.out_channels == 3
    assert attention_unet.encoder1.conv_block.conv[0].in_channels == 2
    assert attention_unet.final_conv.out_channels == 3


def test_create_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("bad_model", ExperimentConfig())


def test_create_loss_fn_uses_configured_loss_type() -> None:
    config = ExperimentConfig()

    config.training.loss_type = "DiceCELoss"
    assert isinstance(create_loss_fn(config), DiceCELoss)

    config.training.loss_type = "DiceLoss"
    assert isinstance(create_loss_fn(config), DiceLoss)

    config.training.loss_type = "TverskyLoss"
    assert isinstance(create_loss_fn(config), TverskyLoss)


def test_create_loss_fn_rejects_unknown_loss() -> None:
    config = ExperimentConfig()
    config.training.loss_type = "BadLoss"

    with pytest.raises(ValueError, match="Unknown loss"):
        create_loss_fn(config)


def test_set_seed_makes_torch_randomness_reproducible() -> None:
    set_seed(123)
    first = torch.randn(3)

    set_seed(123)
    second = torch.randn(3)

    assert torch.allclose(first, second)


def test_load_checkpoint_restores_model_weights() -> None:
    root = _fresh_tmp_dir()
    try:
        source_model = torch.nn.Conv3d(4, 4, kernel_size=1)
        target_model = torch.nn.Conv3d(4, 4, kernel_size=1)
        with torch.no_grad():
            source_model.weight.fill_(0.25)
            source_model.bias.fill_(0.5)
            target_model.weight.zero_()
            target_model.bias.zero_()

        checkpoint_path = root / "checkpoint.pth"
        torch.save(
            {
                "epoch": 7,
                "model_state_dict": source_model.state_dict(),
            },
            checkpoint_path,
        )

        checkpoint = load_checkpoint(target_model, checkpoint_path, device="cpu")

        assert checkpoint["epoch"] == 7
        assert torch.allclose(target_model.weight, source_model.weight)
        assert torch.allclose(target_model.bias, source_model.bias)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fresh_tmp_dir() -> Path:
    root = Path("tmp") / f"test_train_script_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
