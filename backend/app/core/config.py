"""Application configuration for the backend API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_CONFIG_PATH = PROJECT_ROOT / "configs" / "app_config.yaml"


class Settings(BaseSettings):
    """Runtime settings loaded from defaults, YAML config, and environment variables."""

    app_name: str = "Brain Tumor Segmentation API"
    app_env: str = "development"
    app_debug: bool = True
    api_prefix: str = "/api"

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    unet_model_path: Path = Path("data/models/unet3d_best.pth")
    attention_unet_model_path: Path = Path("data/models/attention_unet3d_best.pth")
    default_model_name: str = "attention_unet3d"

    max_upload_size_mb: int = 500
    allowed_extensions: tuple[str, ...] = (".nii", ".nii.gz", ".zip")
    upload_dir: Path = Path("tmp/uploads")
    results_dir: Path = Path("tmp/results")
    brats_raw_root: Path = Path("data/raw")
    cleanup_after_hours: int = 24

    segmentation_device: str = "auto"
    cuda_visible_devices: str = "0"

    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "unet_model_path",
        "attention_unet_model_path",
        "upload_dir",
        "results_dir",
        "brats_raw_root",
        mode="before",
    )
    @classmethod
    def _path_from_string(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator("allowed_extensions", "cors_origins", mode="before")
    @classmethod
    def _tuple_from_csv_or_list(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, list):
            return tuple(str(item) for item in value)
        return value

    @property
    def model_paths(self) -> dict[str, Path]:
        return {
            "unet3d": self.unet_model_path,
            "attention_unet3d": self.attention_unet_model_path,
        }

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


def load_yaml_defaults(path: str | Path = APP_CONFIG_PATH) -> dict[str, Any]:
    """Load selected defaults from configs/app_config.yaml if it exists."""
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    server = raw.get("server", {})
    upload = raw.get("upload", {})
    models = raw.get("models", {})
    segmentation = raw.get("segmentation", {})

    available_models = {
        item.get("name"): item
        for item in models.get("available", [])
        if isinstance(item, dict) and item.get("name")
    }

    defaults: dict[str, Any] = {
        "host": server.get("host"),
        "port": server.get("port"),
        "workers": server.get("workers"),
        "max_upload_size_mb": upload.get("max_file_size_mb"),
        "allowed_extensions": upload.get("allowed_extensions"),
        "upload_dir": upload.get("upload_dir"),
        "results_dir": upload.get("results_dir"),
        "brats_raw_root": upload.get("brats_raw_root"),
        "cleanup_after_hours": upload.get("cleanup_after_hours"),
        "default_model_name": models.get("default"),
        "segmentation_device": segmentation.get("device"),
    }

    if "unet3d" in available_models:
        defaults["unet_model_path"] = available_models["unet3d"].get("weight_path")
    if "attention_unet3d" in available_models:
        defaults["attention_unet_model_path"] = available_models["attention_unet3d"].get("weight_path")

    return {key: value for key, value in defaults.items() if value is not None}


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings(**load_yaml_defaults())


settings = get_settings()
