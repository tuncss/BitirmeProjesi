"""Tests for backend file management service."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import InvalidFileFormatError, UploadedFileNotFoundError
from backend.app.services import (
    FileManager,
    UPLOAD_MANIFEST_FILENAME,
    infer_modality_from_filename,
    normalized_extension,
)


def test_normalized_extension_handles_medical_suffixes() -> None:
    assert normalized_extension("case_t1.nii.gz") == ".nii.gz"
    assert normalized_extension("case_t1.nii") == ".nii"
    assert normalized_extension("case.zip") == ".zip"
    assert normalized_extension("notes.txt") == ".txt"


def test_infer_modality_from_filename_does_not_confuse_t1ce_with_t1() -> None:
    assert infer_modality_from_filename("BraTS2021_00000_t1ce.nii.gz") == "t1ce"
    assert infer_modality_from_filename("BraTS2021_00000_t1.nii.gz") == "t1"
    assert infer_modality_from_filename("BraTS2021_00000_t2.nii.gz") == "t2"
    assert infer_modality_from_filename("BraTS2021_00000_flair.nii.gz") == "flair"
    assert infer_modality_from_filename("unknown.nii.gz") is None


def test_create_session_and_save_file() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        session_id = manager.create_upload_session()

        saved = manager.save_file(session_id, "case_t1.nii.gz", b"content")

        assert saved.exists()
        assert saved.read_bytes() == b"content"
        assert saved.parent == manager.upload_dir / session_id
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_save_file_records_original_modality_filename() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        session_id = manager.create_upload_session()

        manager.save_file(session_id, "BraTS2021_00042_t1.nii.gz", b"content")

        session_dir = manager.get_upload_session_dir(session_id)
        manifest_path = session_dir / UPLOAD_MANIFEST_FILENAME
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
            "t1": "BraTS2021_00042_t1.nii.gz"
        }
        assert manager.get_original_filenames(session_id) == {
            "t1": "BraTS2021_00042_t1.nii.gz"
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_save_file_rejects_unsupported_extension() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)

        with pytest.raises(InvalidFileFormatError):
            manager.save_file("session", "readme.txt", b"bad")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_extract_zip_saves_supported_members_flattened() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        session_id = manager.create_upload_session()
        zip_path = manager.get_upload_session_dir(session_id) / "modalities.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("nested/case_t1.nii.gz", b"t1")
            archive.writestr("../../case_t2.nii.gz", b"t2")

        extracted = manager.extract_zip(session_id, zip_path)

        assert sorted(path.name for path in extracted) == ["case_t1.nii.gz", "case_t2.nii.gz"]
        assert (manager.get_upload_session_dir(session_id) / "case_t2.nii.gz").exists()
        assert not (root / "case_t2.nii.gz").exists()
        assert manager.get_original_filenames(session_id) == {
            "t1": "case_t1.nii.gz",
            "t2": "case_t2.nii.gz",
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_extract_zip_rejects_unsupported_members() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        session_id = manager.create_upload_session()
        zip_path = manager.get_upload_session_dir(session_id) / "modalities.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("notes.txt", b"bad")

        with pytest.raises(InvalidFileFormatError):
            manager.extract_zip(session_id, zip_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_identify_modalities_maps_required_files() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        session_id = manager.create_upload_session()
        session_dir = manager.get_upload_session_dir(session_id)
        for modality in ["t1", "t1ce", "t2", "flair"]:
            (session_dir / f"BraTS2021_00000_{modality}.nii.gz").write_bytes(b"placeholder")

        modalities = manager.identify_modalities(session_id)

        assert list(modalities) == ["t1", "t1ce", "t2", "flair"]
        assert modalities["t1ce"].name.endswith("_t1ce.nii.gz")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_get_original_filenames_falls_back_when_manifest_is_missing() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        session_id = manager.create_upload_session()
        session_dir = manager.get_upload_session_dir(session_id)
        for modality in ["t1", "t1ce", "t2", "flair"]:
            (session_dir / f"BraTS2021_00000_{modality}.nii.gz").write_bytes(b"placeholder")

        assert manager.get_original_filenames(session_id) == {
            "t1": "BraTS2021_00000_t1.nii.gz",
            "t1ce": "BraTS2021_00000_t1ce.nii.gz",
            "t2": "BraTS2021_00000_t2.nii.gz",
            "flair": "BraTS2021_00000_flair.nii.gz",
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_identify_modalities_rejects_duplicates_and_missing() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        duplicate_session = manager.create_upload_session()
        duplicate_dir = manager.get_upload_session_dir(duplicate_session)
        (duplicate_dir / "a_t1.nii.gz").write_bytes(b"")
        (duplicate_dir / "b_t1.nii.gz").write_bytes(b"")

        with pytest.raises(ValueError, match="Duplicate modality"):
            manager.identify_modalities(duplicate_session)

        missing_session = manager.create_upload_session()
        (manager.get_upload_session_dir(missing_session) / "case_t1.nii.gz").write_bytes(b"")
        with pytest.raises(ValueError, match="Missing modalities"):
            manager.identify_modalities(missing_session)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validate_nifti_files_checks_completeness_and_shapes() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        modality_map = _write_nifti_modalities(root)

        assert manager.validate_nifti_files(modality_map) is True

        modality_map["flair"] = _write_nifti(root / "bad_flair.nii.gz", shape=(5, 4, 4))
        with pytest.raises(ValueError, match="shapes must match"):
            manager.validate_nifti_files(modality_map)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validate_nifti_files_wraps_invalid_nifti() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        modality_map = _write_nifti_modalities(root)
        Path(modality_map["t2"]).write_bytes(b"not nifti")

        with pytest.raises(ValueError, match="Invalid NIfTI file for t2"):
            manager.validate_nifti_files(modality_map)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_get_session_and_result_paths_validate_ids() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)

        with pytest.raises(ValueError, match="Invalid session_id"):
            manager.get_upload_session_dir("../bad", create=True)

        with pytest.raises(UploadedFileNotFoundError):
            manager.get_upload_session_dir("missing")

        result_dir = manager.create_result_dir("task123")
        assert result_dir.exists()
        assert manager.get_result_path("task123").name == "segmentation.nii.gz"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cleanup_old_files_removes_stale_directories() -> None:
    root = _fresh_tmp_dir()
    try:
        manager = _manager(root)
        old_upload = manager.upload_dir / "old_upload"
        fresh_upload = manager.upload_dir / "fresh_upload"
        old_result = manager.results_dir / "old_result"
        for directory in [old_upload, fresh_upload, old_result]:
            directory.mkdir(parents=True, exist_ok=True)

        old_time = time.time() - 3 * 3600
        os.utime(old_upload, (old_time, old_time))
        os.utime(old_result, (old_time, old_time))

        removed = manager.cleanup_old_files(max_age_hours=1)

        assert old_upload in removed
        assert old_result in removed
        assert not old_upload.exists()
        assert not old_result.exists()
        assert fresh_upload.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _manager(root: Path) -> FileManager:
    return FileManager(
        Settings(
            upload_dir=root / "uploads",
            results_dir=root / "results",
            allowed_extensions=(".nii", ".nii.gz", ".zip"),
        )
    )


def _write_nifti_modalities(root: Path, shape: tuple[int, int, int] = (4, 4, 4)) -> dict[str, str]:
    return {
        modality: _write_nifti(root / f"case_{modality}.nii.gz", shape=shape)
        for modality in ["t1", "t1ce", "t2", "flair"]
    }


def _write_nifti(path: Path, shape: tuple[int, int, int]) -> str:
    data = np.zeros(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return str(path)


def _fresh_tmp_dir() -> Path:
    root = Path("tmp") / f"test_file_manager_{uuid.uuid4().hex}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
