"""Tests for typed training configuration objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.training import ExperimentConfig


def test_experiment_config_defaults() -> None:
    config = ExperimentConfig()

    assert config.model_name == "unet3d"
    assert config.seed == 42
    assert config.data.dataset == "BraTS2021"
    assert config.data.input_shape == [128, 128, 128]
    assert config.data.num_classes == 4
    assert config.training.epochs == 300
    assert config.training.batch_size == 2
    assert config.training.learning_rate == 1e-4
    assert config.training.optimizer == "AdamW"
    assert config.training.scheduler_type == "CosineAnnealingWarmRestarts"
    assert config.training.loss_type == "DiceCELoss"
    assert config.training.use_mixed_precision is True
    assert config.checkpoint.save_dir == "data/models"
    assert config.checkpoint.mode == "max"
    assert config.logging.tensorboard_dir == "logs/tensorboard"
    assert config.evaluation.compute_hd95_every == 10


def test_experiment_config_loads_existing_yaml() -> None:
    config = ExperimentConfig.from_yaml("configs/training_config.yaml")

    assert config.data.dataset == "BraTS2021"
    assert config.data.processed_dir == "data/processed"
    assert config.data.modalities == ["t1", "t1ce", "t2", "flair"]
    assert config.training.models == ["unet3d", "attention_unet3d"]
    assert config.training.epochs == 300
    assert config.training.scheduler_type == "CosineAnnealingWarmRestarts"
    assert config.training.scheduler_T_0 == 50
    assert config.training.scheduler_T_mult == 2
    assert config.training.loss_type == "DiceCELoss"
    assert config.training.dice_weight == 1.0
    assert config.training.ce_weight == 1.0
    assert config.training.early_stopping_patience == 50
    assert config.training.early_stopping_min_delta == 0.001
    assert config.training.early_stopping_monitor == "val_dice_mean"
    assert config.checkpoint.save_dir == "data/models"
    assert config.checkpoint.save_best_only is True
    assert config.checkpoint.monitor == "val_dice_mean"
    assert config.logging.log_interval == 10
    assert config.evaluation.compute_hd95_every == 10
    assert config.evaluation.target_dice == {"WT": 0.88, "TC": 0.82, "ET": 0.78}


def test_experiment_config_to_dict_is_plain_nested_dict() -> None:
    config = ExperimentConfig()
    config.model_name = "attention_unet3d"

    data = config.to_dict()

    assert data["model_name"] == "attention_unet3d"
    assert data["training"]["epochs"] == 300
    assert data["checkpoint"]["monitor"] == "val_dice_mean"
    assert isinstance(data["data"]["modalities"], list)


def test_experiment_config_accepts_checkpoint_alias() -> None:
    config_path = Path("tmp/test_config_checkpoint_alias.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "checkpointing": {
                    "save_dir": "custom/models",
                    "save_best_only": False,
                    "monitor": "val_loss",
                }
            }
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(config_path)

    assert config.checkpoint.save_dir == "custom/models"
    assert config.checkpoint.save_best_only is False
    assert config.checkpoint.monitor == "val_loss"
    assert config.checkpoint.mode == "max"
    config_path.unlink()


def test_experiment_config_ignores_unknown_yaml_keys() -> None:
    config_path = Path("tmp/test_config_unknown_keys.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "unknown_section": {"value": 1},
                "training": {
                    "epochs": 2,
                    "scheduler": {"type": "none"},
                    "unknown_training_key": "ignored",
                },
            }
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(config_path)

    assert config.training.epochs == 2
    assert config.training.scheduler_type == "none"
    assert not hasattr(config.training, "unknown_training_key")
    config_path.unlink()
