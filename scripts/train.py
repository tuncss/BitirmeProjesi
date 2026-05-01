"""Command-line entry point for training segmentation models."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models import AttentionUNet3D, DiceCELoss, DiceLoss, TverskyLoss, UNet3D
from src.training import ExperimentConfig, Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_name: str, config: ExperimentConfig) -> torch.nn.Module:
    """Create a model instance from its CLI/config name."""
    in_channels = len(config.data.modalities)
    num_classes = config.data.num_classes

    if model_name == "unet3d":
        return UNet3D(in_channels=in_channels, num_classes=num_classes)
    if model_name == "attention_unet3d":
        return AttentionUNet3D(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def create_loss_fn(config: ExperimentConfig):
    """Create the configured segmentation loss function."""
    loss_type = config.training.loss_type
    if loss_type == "DiceCELoss":
        return DiceCELoss(
            dice_weight=config.training.dice_weight,
            ce_weight=config.training.ce_weight,
        )
    if loss_type == "DiceLoss":
        return DiceLoss()
    if loss_type == "TverskyLoss":
        return TverskyLoss()
    raise ValueError(f"Unknown loss: {loss_type}")


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply command-line overrides to a loaded experiment config."""
    config.model_name = args.model
    config.seed = args.seed

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name

    return config


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: str) -> dict:
    """Load model weights from a training checkpoint and return the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brain tumor segmentation training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["unet3d", "attention_unet3d"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume model weights from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name for TensorBoard",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    config = apply_cli_overrides(config, args)
    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            "GPU Memory: {:.1f} GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        logger.warning("CUDA not available. Training on CPU will be very slow.")

    loaders = get_dataloaders(config.to_dict(), seed=config.seed)
    model = create_model(args.model, config)
    params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Model: {args.model}, Parameters: {params:,}")

    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        logger.info(f"Resumed model weights from: {args.resume} (epoch {checkpoint['epoch']})")

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        loss_fn=create_loss_fn(config),
        device=device,
    )
    result = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info(f"Model: {result['model_name']}")
    logger.info(f"Best Epoch: {result['best_epoch']}")
    logger.info(f"Best {config.checkpoint.monitor}: {result['best_metric']:.4f}")
    logger.info(f"Total Epochs: {result['total_epochs']}")
    logger.info(f"Best model saved: {Path(config.checkpoint.save_dir) / f'{args.model}_best.pth'}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
