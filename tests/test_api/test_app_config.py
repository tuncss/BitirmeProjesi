"""Tests for backend app configuration and health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.config import Settings, load_yaml_defaults
from backend.app.core.exceptions import ModelNotFoundError
from backend.app.main import create_app


def test_load_yaml_defaults_reads_app_config() -> None:
    defaults = load_yaml_defaults("configs/app_config.yaml")

    assert defaults["host"] == "0.0.0.0"
    assert defaults["port"] == 8000
    assert defaults["max_upload_size_mb"] == 500
    assert defaults["brats_raw_root"] == "data/raw"
    assert defaults["default_model_name"] == "attention_unet3d"
    assert defaults["unet_model_path"] == "data/models/unet3d_best.pth"
    assert defaults["attention_unet_model_path"] == "data/models/attention_unet3d_best.pth"


def test_settings_supports_env_style_overrides() -> None:
    settings = Settings(
        app_env="test",
        app_debug=False,
        unet_model_path="model_artifacts/unet.pth",
        brats_raw_root="data/raw",
        allowed_extensions=".nii,.nii.gz",
        cors_origins="http://example.test,http://localhost:5173",
    )

    assert settings.app_env == "test"
    assert settings.app_debug is False
    assert settings.unet_model_path.as_posix() == "model_artifacts/unet.pth"
    assert settings.brats_raw_root.as_posix() == "data/raw"
    assert settings.allowed_extensions == (".nii", ".nii.gz")
    assert settings.cors_origins == ("http://example.test", "http://localhost:5173")
    assert settings.max_upload_size_bytes == 500 * 1024 * 1024


def test_health_endpoint_returns_backend_status() -> None:
    app = create_app(Settings(app_env="test"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["environment"] == "test"
    assert payload["device"] in {"cpu", "cuda"}
    assert "unet3d" in payload["models"]
    assert "attention_unet3d" in payload["models"]


def test_openapi_and_docs_are_available() -> None:
    app = create_app(Settings(app_env="test"))
    client = TestClient(app)

    openapi_response = client.get("/openapi.json")
    docs_response = client.get("/docs")

    assert openapi_response.status_code == 200
    assert openapi_response.json()["info"]["title"] == "Brain Tumor Segmentation API"
    assert docs_response.status_code == 200


def test_api_errors_use_consistent_envelope() -> None:
    app = create_app(Settings(app_env="test"))

    @app.get("/raise-model-error")
    async def raise_model_error():
        raise ModelNotFoundError("bad_model")

    client = TestClient(app)
    response = client.get("/raise-model-error")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "MODEL_NOT_FOUND",
            "message": "Model not found: bad_model",
            "details": None,
        },
        "meta": {
            "path": "/raise-model-error",
        },
    }
