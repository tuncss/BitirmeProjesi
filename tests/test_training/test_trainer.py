"""Tests for the training loop orchestration."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import ExperimentConfig, Trainer


def test_trainer_runs_one_epoch_and_saves_best_checkpoint() -> None:
    root = _fresh_tmp_dir()
    try:
        config = _make_config(root, epochs=1)
        model = nn.Conv3d(4, 4, kernel_size=1)
        initial_weight = model.weight.detach().clone()
        loader = _make_loader()

        trainer = Trainer(
            model=model,
            config=config,
            train_loader=loader,
            val_loader=loader,
            loss_fn=nn.CrossEntropyLoss(),
            device="cpu",
        )
        result = trainer.train()

        assert result["best_epoch"] == 1
        assert result["total_epochs"] == 1
        assert result["model_name"] == "tiny_unet"
        assert result["best_metric"] > 0.0
        assert trainer.use_amp is False
        assert not torch.allclose(model.weight.detach(), initial_weight)

        checkpoint_path = root / "models" / "tiny_unet_best.pth"
        metrics_path = root / "models" / "tiny_unet_best_metrics.json"
        assert checkpoint_path.exists()
        assert metrics_path.exists()

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["config"]["model_name"] == "tiny_unet"
        assert checkpoint["metrics"]["mean_dice"] == pytest.approx(result["best_metric"])
        assert any((root / "logs").rglob("events.out.tfevents.*"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_trainer_resolves_val_dice_mean_monitor_alias() -> None:
    metrics = {"mean_dice": 0.72}

    assert Trainer._get_monitor_value(metrics, "val_dice_mean") == 0.72


def test_trainer_early_stopping_uses_patience(monkeypatch) -> None:
    root = _fresh_tmp_dir()
    try:
        config = _make_config(root, epochs=5)
        config.training.early_stopping_patience = 1
        config.training.early_stopping_min_delta = 0.001
        model = nn.Conv3d(4, 4, kernel_size=1)
        loader = _make_loader(num_samples=2)
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=loader,
            val_loader=loader,
            loss_fn=nn.CrossEntropyLoss(),
            device="cpu",
        )

        def constant_validation(epoch: int) -> dict[str, float]:
            return {
                "mean_dice": 0.5,
                "WT_dice": 0.5,
                "TC_dice": 0.5,
                "ET_dice": 0.5,
            }

        monkeypatch.setattr(trainer, "_validate", constant_validation)

        result = trainer.train()

        assert result["best_epoch"] == 1
        assert result["best_metric"] == 0.5
        assert result["total_epochs"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_trainer_computes_hd95_only_on_configured_interval(monkeypatch) -> None:
    root = _fresh_tmp_dir()
    try:
        config = _make_config(root, epochs=1)
        config.evaluation.compute_hd95_every = 2
        loader = _make_loader(num_samples=2)
        trainer = Trainer(
            model=nn.Conv3d(4, 4, kernel_size=1),
            config=config,
            train_loader=loader,
            val_loader=loader,
            loss_fn=nn.CrossEntropyLoss(),
            device="cpu",
        )
        calls: list[bool] = []

        def fake_compute_brats_metrics(pred_mask, gt_mask, include_hd95: bool = True):
            calls.append(include_hd95)
            return {
                "WT_dice": 1.0,
                "WT_iou": 1.0,
                "TC_dice": 1.0,
                "TC_iou": 1.0,
                "ET_dice": 1.0,
                "ET_iou": 1.0,
                "mean_dice": 1.0,
            }

        monkeypatch.setattr(
            "src.training.trainer.compute_brats_metrics",
            fake_compute_brats_metrics,
        )

        trainer._validate(epoch=1)
        assert calls == [False, False]

        calls.clear()
        trainer._validate(epoch=2)
        assert calls == [True, True]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_trainer_raises_for_unknown_optimizer() -> None:
    root = _fresh_tmp_dir()
    try:
        config = _make_config(root)
        config.training.optimizer = "SGD"

        with pytest.raises(ValueError, match="Unknown optimizer"):
            Trainer(
                model=nn.Conv3d(4, 4, kernel_size=1),
                config=config,
                train_loader=_make_loader(),
                val_loader=_make_loader(),
                loss_fn=nn.CrossEntropyLoss(),
                device="cpu",
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_trainer_raises_when_monitor_metric_is_missing() -> None:
    with pytest.raises(KeyError, match="Monitor metric"):
        Trainer._get_monitor_value({"mean_dice": 0.7}, "val_loss")


def _make_config(root: Path, epochs: int = 1) -> ExperimentConfig:
    config = ExperimentConfig()
    config.model_name = "tiny_unet"
    config.experiment_name = f"test_{root.name}"
    config.training.epochs = epochs
    config.training.batch_size = 2
    config.training.learning_rate = 1e-2
    config.training.scheduler_type = "none"
    config.training.use_mixed_precision = True
    config.checkpoint.save_dir = str(root / "models")
    config.checkpoint.monitor = "val_dice_mean"
    config.checkpoint.mode = "max"
    config.logging.tensorboard_dir = str(root / "logs")
    config.logging.log_interval = 1
    return config


def _make_loader(num_samples: int = 4, batch_size: int = 2) -> DataLoader:
    generator = torch.Generator().manual_seed(42)
    images = torch.randn(num_samples, 4, 6, 6, 6, generator=generator)
    mask_pattern = torch.arange(6 * 6 * 6).reshape(6, 6, 6) % 4
    masks = mask_pattern.unsqueeze(0).repeat(num_samples, 1, 1, 1).long()
    return DataLoader(TensorDataset(images, masks), batch_size=batch_size)


def _fresh_tmp_dir() -> Path:
    root = Path("tmp") / f"test_trainer_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
