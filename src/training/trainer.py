"""Training loop orchestration for segmentation models."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.training.config import ExperimentConfig
from src.training.evaluator import MetricTracker, compute_brats_metrics


class Trainer:
    """Orchestrates training, validation, checkpointing, and TensorBoard logging."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader,
        val_loader,
        loss_fn,
        device: str = "cuda",
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.device_type = self.device.type

        self.model = model.to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.use_amp = bool(config.training.use_mixed_precision and self.device_type == "cuda")
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)

        self.monitor_name = config.checkpoint.monitor or config.training.early_stopping_monitor
        self.monitor_mode = config.checkpoint.mode
        self.best_metric = -math.inf if self.monitor_mode == "max" else math.inf
        self.patience_counter = 0
        self.best_epoch = 0

        exp_name = config.experiment_name or f"{config.model_name}_{int(time.time())}"
        self.writer = SummaryWriter(log_dir=str(Path(config.logging.tensorboard_dir) / exp_name))

        self.checkpoint_dir = Path(config.checkpoint.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config.training
        if cfg.optimizer == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        if cfg.optimizer == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
            )
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _create_scheduler(self):
        cfg = self.config.training
        if cfg.scheduler_type == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cfg.scheduler_T_0,
                T_mult=cfg.scheduler_T_mult,
            )
        if cfg.scheduler_type in ("none", "None", None):
            return None
        raise ValueError(f"Unknown scheduler: {cfg.scheduler_type}")

    def train(self) -> dict[str, Any]:
        """Run the full training loop and return a compact training summary."""
        logger.info(f"Training started: {self.config.model_name}")
        logger.info(
            "Epochs: {}, Batch size: {}, LR: {}",
            self.config.training.epochs,
            self.config.training.batch_size,
            self.config.training.learning_rate,
        )
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Device: {self.device}")

        self.writer.add_hparams(
            _flatten_hparams(self.config.to_dict()),
            {"hparams/placeholder": 0.0},
        )

        final_epoch = 0
        for epoch in range(1, self.config.training.epochs + 1):
            final_epoch = epoch
            train_loss = self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", current_lr, epoch)
            self.writer.add_scalar("train/loss", train_loss, epoch)
            for key, value in val_metrics.items():
                if math.isfinite(float(value)):
                    self.writer.add_scalar(f"val/{key}", float(value), epoch)

            monitor_value = self._get_monitor_value(val_metrics, self.monitor_name)
            logger.info(
                "Epoch {}/{} | Train Loss: {:.4f} | {}: {:.4f} | WT: {:.4f} | TC: {:.4f} | ET: {:.4f} | LR: {:.6f}",
                epoch,
                self.config.training.epochs,
                train_loss,
                self.monitor_name,
                monitor_value,
                val_metrics.get("WT_dice", 0.0),
                val_metrics.get("TC_dice", 0.0),
                val_metrics.get("ET_dice", 0.0),
                current_lr,
            )

            improved = self._is_improvement(monitor_value)
            if improved:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, suffix="best")
                logger.info(f"New best model: {self.monitor_name}={monitor_value:.4f}")
            else:
                self.patience_counter += 1

            if not self.config.checkpoint.save_best_only:
                self._save_checkpoint(epoch, val_metrics, suffix=f"epoch_{epoch}")

            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                break

        self.writer.close()
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "total_epochs": final_epoch,
            "model_name": self.config.model_name,
        }

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, (images, masks) in enumerate(progress):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            num_batches += 1
            progress.set_postfix({"loss": f"{loss_value:.4f}"})

            if batch_idx % self.config.logging.log_interval == 0:
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/batch_loss", loss_value, global_step)

        if num_batches == 0:
            raise ValueError("train_loader produced no batches")
        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()
        num_batches = 0
        include_hd95 = self._should_compute_hd95(epoch)

        progress = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, masks in progress:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                predictions = self.model(images)

            pred_labels = predictions.argmax(dim=1)
            for sample_idx in range(pred_labels.shape[0]):
                metrics = compute_brats_metrics(
                    pred_labels[sample_idx].cpu().numpy(),
                    masks[sample_idx].cpu().numpy(),
                    include_hd95=include_hd95,
                )
                tracker.update(metrics)
            num_batches += 1

        if num_batches == 0:
            raise ValueError("val_loader produced no batches")
        return tracker.compute()

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float], suffix: str) -> None:
        model_name = self.config.model_name
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "metrics": metrics,
            "config": self.config.to_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }

        checkpoint_path = self.checkpoint_dir / f"{model_name}_{suffix}.pth"
        torch.save(checkpoint, checkpoint_path)

        metrics_path = self.checkpoint_dir / f"{model_name}_{suffix}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump({"epoch": epoch, **metrics}, file, indent=2, allow_nan=True)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _is_improvement(self, value: float) -> bool:
        min_delta = self.config.training.early_stopping_min_delta
        if self.monitor_mode == "max":
            return value > self.best_metric + min_delta
        if self.monitor_mode == "min":
            return value < self.best_metric - min_delta
        raise ValueError(f"Unknown checkpoint mode: {self.monitor_mode}")

    def _should_compute_hd95(self, epoch: int) -> bool:
        interval = self.config.evaluation.compute_hd95_every
        return interval > 0 and epoch % interval == 0

    @staticmethod
    def _get_monitor_value(metrics: dict[str, float], monitor: str) -> float:
        candidates = [monitor]
        if monitor.startswith("val_"):
            candidates.append(monitor.removeprefix("val_"))
        if monitor == "val_dice_mean":
            candidates.append("mean_dice")
        if monitor.endswith("_dice_mean"):
            candidates.append("mean_dice")

        for candidate in candidates:
            if candidate in metrics:
                return float(metrics[candidate])

        available = ", ".join(sorted(metrics))
        raise KeyError(f"Monitor metric '{monitor}' not found. Available metrics: {available}")


def _flatten_hparams(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_hparams(value, name))
        elif isinstance(value, (list, tuple)):
            flattened[name] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool, torch.Tensor)):
            flattened[name] = value
        elif value is None:
            flattened[name] = "None"
        else:
            flattened[name] = str(value)
    return flattened
