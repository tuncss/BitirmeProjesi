"""Shared fixtures for backend API tests."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


@pytest.fixture
def api_tmp_root() -> Path:
    root = Path("tmp") / f"test_api_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def api_settings(api_tmp_root: Path) -> Settings:
    return Settings(
        app_env="test",
        upload_dir=api_tmp_root / "uploads",
        results_dir=api_tmp_root / "results",
        default_model_name="attention_unet3d",
        unet_model_path=api_tmp_root / "unet.pth",
        attention_unet_model_path=api_tmp_root / "attention.pth",
    )


@pytest.fixture
def api_client(api_settings: Settings) -> TestClient:
    return TestClient(create_app(api_settings))


@pytest.fixture
def valid_upload_session(api_tmp_root: Path) -> str:
    session_id = uuid.uuid4().hex
    session_dir = api_tmp_root / "uploads" / session_id
    session_dir.mkdir(parents=True)
    for modality in ["t1", "t1ce", "t2", "flair"]:
        (session_dir / f"BraTS2021_00000_{modality}.nii.gz").write_bytes(make_nifti_bytes(api_tmp_root, modality))
    return session_id


def make_nifti_bytes(root: Path, name: str, shape: tuple[int, int, int] = (4, 4, 4)) -> bytes:
    path = root / f"{name}_{uuid.uuid4().hex}.nii.gz"
    data = np.zeros(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    content = path.read_bytes()
    path.unlink()
    return content

