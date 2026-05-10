"""Tests for BraTS subject lookup helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.services.brats_lookup import extract_subject_id, resolve_gt_path


def test_extract_subject_id_returns_shared_brats_id() -> None:
    modality_paths = {
        "t1": Path("uploads/session/BraTS2021_00042_t1.nii.gz"),
        "t1ce": Path("uploads/session/BraTS2021_00042_t1ce.nii.gz"),
        "t2": Path("uploads/session/BraTS2021_00042_t2.nii.gz"),
        "flair": Path("uploads/session/BraTS2021_00042_flair.nii.gz"),
    }

    assert extract_subject_id(modality_paths) == "BraTS2021_00042"


def test_extract_subject_id_is_case_insensitive() -> None:
    modality_paths = {
        "t1": "BraTS2021_00042_T1.nii.gz",
        "t1ce": "BraTS2021_00042_T1CE.nii.gz",
        "t2": "brats2021_00042_t2.nii.gz",
        "flair": "BRATS2021_00042_FLAIR.NII.GZ",
    }

    assert extract_subject_id(modality_paths) == "BraTS2021_00042"


def test_extract_subject_id_returns_none_for_mixed_subject_ids() -> None:
    modality_paths = {
        "t1": "BraTS2021_00042_t1.nii.gz",
        "t1ce": "BraTS2021_00042_t1ce.nii.gz",
        "t2": "BraTS2021_00099_t2.nii.gz",
        "flair": "BraTS2021_00042_flair.nii.gz",
    }

    assert extract_subject_id(modality_paths) is None


def test_extract_subject_id_returns_none_for_non_brats_filenames() -> None:
    modality_paths = {
        "t1": "patient_a_t1.nii.gz",
        "t1ce": "patient_a_t1ce.nii.gz",
        "t2": "patient_a_t2.nii.gz",
        "flair": "patient_a_flair.nii.gz",
    }

    assert extract_subject_id(modality_paths) is None


def test_extract_subject_id_returns_none_for_missing_modality() -> None:
    modality_paths = {
        "t1": "BraTS2021_00042_t1.nii.gz",
        "t1ce": "BraTS2021_00042_t1ce.nii.gz",
        "t2": "BraTS2021_00042_t2.nii.gz",
    }

    assert extract_subject_id(modality_paths) is None


def test_extract_subject_id_returns_none_for_filename_modality_mismatch() -> None:
    modality_paths = {
        "t1": "BraTS2021_00042_t2.nii.gz",
        "t1ce": "BraTS2021_00042_t1ce.nii.gz",
        "t2": "BraTS2021_00042_t1.nii.gz",
        "flair": "BraTS2021_00042_flair.nii.gz",
    }

    assert extract_subject_id(modality_paths) is None


def test_resolve_gt_path_returns_existing_segmentation_path(tmp_path: Path) -> None:
    subject_id = "BraTS2021_00042"
    gt_path = tmp_path / "BraTS2021" / subject_id / f"{subject_id}_seg.nii.gz"
    gt_path.parent.mkdir(parents=True)
    gt_path.write_bytes(b"placeholder")

    assert resolve_gt_path(subject_id, tmp_path) == gt_path.resolve()


def test_resolve_gt_path_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert resolve_gt_path("BraTS2021_00042", tmp_path) is None


def test_resolve_gt_path_rejects_invalid_subject_ids(tmp_path: Path) -> None:
    assert resolve_gt_path("../../../etc/passwd", tmp_path) is None
    assert resolve_gt_path("BraTS2021_42", tmp_path) is None
    assert resolve_gt_path("BraTS2020_00042", tmp_path) is None


def test_resolve_gt_path_rejects_symlink_escape(tmp_path: Path) -> None:
    subject_id = "BraTS2021_00042"
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_gt = outside_root / f"{subject_id}_seg.nii.gz"
    outside_gt.write_bytes(b"placeholder")

    raw_root = tmp_path / "raw"
    subject_dir = raw_root / "BraTS2021" / subject_id
    subject_dir.parent.mkdir(parents=True)
    try:
        subject_dir.symlink_to(outside_root, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is not supported in this environment.")

    assert resolve_gt_path(subject_id, raw_root) is None
