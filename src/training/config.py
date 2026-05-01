"""Typed configuration objects for model training experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    dataset: str = "BraTS2021"
    data_dir: str = "data/raw/BraTS2021"
    processed_dir: str = "data/processed"
    input_shape: list[int] = field(default_factory=lambda: [128, 128, 128])
    num_classes: int = 4
    modalities: list[str] = field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    split: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})


@dataclass
class TrainingConfig:
    models: list[str] = field(default_factory=lambda: ["unet3d", "attention_unet3d"])
    epochs: int = 300
    batch_size: int = 2
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    weight_decay: float = 1e-5
    scheduler_type: str = "CosineAnnealingWarmRestarts"
    scheduler_T_0: int = 50
    scheduler_T_mult: int = 2
    loss_type: str = "DiceCELoss"
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_dice_mean"
    use_mixed_precision: bool = True


@dataclass
class CheckpointConfig:
    save_dir: str = "data/models"
    save_best_only: bool = True
    monitor: str = "val_dice_mean"
    mode: str = "max"


@dataclass
class LoggingConfig:
    tensorboard_dir: str = "logs/tensorboard"
    log_interval: int = 10
    save_visualization_every: int = 5


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["dice", "iou", "hausdorff95"])
    compute_hd95_every: int = 10
    regions: dict[str, str] = field(
        default_factory=lambda: {
            "WT": "Whole Tumor",
            "TC": "Tumor Core",
            "ET": "Enhancing Tumor",
        }
    )
    target_dice: dict[str, float] = field(
        default_factory=lambda: {
            "WT": 0.88,
            "TC": 0.82,
            "ET": 0.78,
        }
    )


@dataclass
class ExperimentConfig:
    """Complete configuration for a training experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model_name: str = "unet3d"
    experiment_name: str = ""
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load experiment configuration from the project YAML format."""
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        config = cls()
        config.data = _update_dataclass(config.data, raw.get("data", {}))
        config.training = _training_from_raw(raw.get("training", {}), config.training)
        config.checkpoint = _checkpoint_from_raw(raw, config.checkpoint)
        config.logging = _update_dataclass(config.logging, raw.get("logging", {}))
        config.evaluation = _update_dataclass(config.evaluation, raw.get("evaluation", {}))

        if "model_name" in raw:
            config.model_name = str(raw["model_name"])
        if "experiment_name" in raw:
            config.experiment_name = str(raw["experiment_name"])
        if "seed" in raw:
            config.seed = int(raw["seed"])

        return config

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation for logging and checkpoints."""
        return asdict(self)


T = TypeVar("T")


def _training_from_raw(raw: dict[str, Any], default: TrainingConfig) -> TrainingConfig:
    flattened = dict(raw)
    scheduler = flattened.pop("scheduler", {}) or {}
    loss = flattened.pop("loss", {}) or {}
    early_stopping = flattened.pop("early_stopping", {}) or {}

    mapped = {
        **flattened,
        "scheduler_type": scheduler.get("type", default.scheduler_type),
        "scheduler_T_0": scheduler.get("T_0", default.scheduler_T_0),
        "scheduler_T_mult": scheduler.get("T_mult", default.scheduler_T_mult),
        "loss_type": loss.get("type", default.loss_type),
        "dice_weight": loss.get("dice_weight", default.dice_weight),
        "ce_weight": loss.get("ce_weight", default.ce_weight),
        "early_stopping_patience": early_stopping.get("patience", default.early_stopping_patience),
        "early_stopping_min_delta": early_stopping.get(
            "min_delta",
            default.early_stopping_min_delta,
        ),
        "early_stopping_monitor": early_stopping.get("monitor", default.early_stopping_monitor),
    }
    return _update_dataclass(default, mapped)


def _checkpoint_from_raw(raw: dict[str, Any], default: CheckpointConfig) -> CheckpointConfig:
    checkpoint_raw = raw.get("checkpoint", raw.get("checkpointing", {}))
    return _update_dataclass(default, checkpoint_raw or {})


def _update_dataclass(instance: T, values: dict[str, Any]) -> T:
    if not is_dataclass(instance):
        raise TypeError(f"Expected a dataclass instance, got {type(instance)!r}")

    allowed_fields = {field.name for field in fields(instance)}
    filtered = {key: value for key, value in values.items() if key in allowed_fields}
    for key, value in filtered.items():
        setattr(instance, key, value)
    return instance
