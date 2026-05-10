"""BraTS dataset lookup helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from backend.app.services.file_manager import REQUIRED_MODALITIES


BRATS_MODALITY_FILENAME_RE = re.compile(
    r"^BraTS2021_(?P<digits>\d{5})_(?P<modality>t1|t1ce|t2|flair)\.nii\.gz$",
    re.IGNORECASE,
)
BRATS_SUBJECT_ID_RE = re.compile(r"^BraTS2021_\d{5}$")


def extract_subject_id(modality_paths: Mapping[str, str | Path]) -> str | None:
    """Return the shared BraTS subject id when all required modality filenames match.

    The subject id is derived from the filename only, not from parent directories.
    A complete set of t1, t1ce, t2, and flair files must all match the
    BraTS2021_XXXXX_<modality>.nii.gz convention and share the same id.
    """
    if not all(modality in modality_paths for modality in REQUIRED_MODALITIES):
        return None

    subject_ids: set[str] = set()
    for modality in REQUIRED_MODALITIES:
        filename = Path(modality_paths[modality]).name
        match = BRATS_MODALITY_FILENAME_RE.fullmatch(filename)
        if match is None:
            return None

        filename_modality = match.group("modality").lower()
        if filename_modality != modality:
            return None

        subject_ids.add(f"BraTS2021_{match.group('digits')}")

    if len(subject_ids) != 1:
        return None
    return subject_ids.pop()


def resolve_gt_path(subject_id: str, raw_root: Path) -> Path | None:
    """Resolve a BraTS ground-truth segmentation path under raw_root safely."""
    if BRATS_SUBJECT_ID_RE.fullmatch(subject_id) is None:
        return None

    root = Path(raw_root).resolve()
    gt_path = root / "BraTS2021" / subject_id / f"{subject_id}_seg.nii.gz"
    resolved_gt_path = gt_path.resolve()

    if root != resolved_gt_path and root not in resolved_gt_path.parents:
        return None
    if not resolved_gt_path.is_file():
        return None
    return resolved_gt_path
