"""BraTS 2021 download guidance and dataset verification.

This script does not download the dataset automatically unless Kaggle CLI is
configured by the user. Its main purpose for the project is to verify that a
manually downloaded BraTS 2021 Task 1 dataset is organized correctly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np


EXPECTED_MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")
EXPECTED_CASE_COUNT = 1251
EXPECTED_SHAPE = (240, 240, 155)
EXPECTED_SEG_LABELS = {0, 1, 2, 4}


@dataclass
class VerificationReport:
    root: Path
    case_count: int = 0
    nifti_count: int = 0
    zero_byte_files: list[Path] = field(default_factory=list)
    top_level_files: list[Path] = field(default_factory=list)
    bad_cases: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    unexpected_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    unexpected_seg_labels: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(
            [
                self.case_count != EXPECTED_CASE_COUNT,
                self.nifti_count != EXPECTED_CASE_COUNT * len(EXPECTED_MODALITIES),
                self.zero_byte_files,
                self.top_level_files,
                self.bad_cases,
                self.unreadable_files,
                self.unexpected_shapes,
                self.unexpected_seg_labels,
            ]
        )


def print_download_guidance(root: Path) -> None:
    print("BraTS 2021 Task 1 dataset guidance")
    print("=" * 38)
    print()
    print("Recommended Kaggle dataset:")
    print("  https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1")
    print()
    print("Optional Kaggle CLI command:")
    print("  kaggle datasets download -d dschettler8845/brats-2021-task1")
    print()
    print("Expected final directory:")
    print(f"  {root}")
    print()
    print("Expected case layout:")
    print("  data/raw/BraTS2021/BraTS2021_00000/")
    print("    BraTS2021_00000_t1.nii.gz")
    print("    BraTS2021_00000_t1ce.nii.gz")
    print("    BraTS2021_00000_t2.nii.gz")
    print("    BraTS2021_00000_flair.nii.gz")
    print("    BraTS2021_00000_seg.nii.gz")
    print()
    print("Run verification:")
    print("  python scripts/download_data.py --verify")


def expected_files_for_case(case_dir: Path) -> set[str]:
    return {f"{case_dir.name}_{modality}.nii.gz" for modality in EXPECTED_MODALITIES}


def verify_dataset(root: Path, quick: bool = False) -> VerificationReport:
    report = VerificationReport(root=root)

    if not root.exists():
        report.bad_cases.append(f"Dataset root does not exist: {root}")
        return report
    if not root.is_dir():
        report.bad_cases.append(f"Dataset root is not a directory: {root}")
        return report

    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    report.case_count = len(case_dirs)
    report.top_level_files = sorted(path for path in root.iterdir() if path.is_file())

    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    report.nifti_count = sum(path.name.endswith((".nii", ".nii.gz")) for path in all_files)
    report.zero_byte_files = [path for path in all_files if path.stat().st_size == 0]

    for case_dir in case_dirs:
        expected = expected_files_for_case(case_dir)
        files = sorted(path.name for path in case_dir.iterdir() if path.is_file())
        missing = sorted(expected.difference(files))
        extra = sorted(set(files).difference(expected))

        if missing or extra or len(files) != len(EXPECTED_MODALITIES):
            report.bad_cases.append(
                f"{case_dir.name}: count={len(files)}, "
                f"missing={missing or '-'}, extra={extra or '-'}"
            )
            continue

        for filename in sorted(expected):
            path = case_dir / filename
            try:
                image = nib.load(str(path))
            except Exception as exc:  # noqa: BLE001 - report file-specific read errors
                report.unreadable_files.append(f"{path}: {exc}")
                continue

            if tuple(image.shape) != EXPECTED_SHAPE:
                report.unexpected_shapes[str(path)] = tuple(image.shape)

            if filename.endswith("_seg.nii.gz"):
                data = np.asanyarray(image.dataobj)
                labels = {int(value) for value in np.unique(data)}
                if not labels.issubset(EXPECTED_SEG_LABELS):
                    report.unexpected_seg_labels[str(path)] = sorted(labels)

        if quick and case_dir == case_dirs[min(9, len(case_dirs) - 1)]:
            break

    return report


def print_report(report: VerificationReport, quick: bool = False) -> None:
    print("BraTS 2021 verification report")
    print("=" * 31)
    print(f"Root: {report.root}")
    print(f"Mode: {'quick sample' if quick else 'full'}")
    print(f"Case directories: {report.case_count}")
    print(f"NIfTI files: {report.nifti_count}")
    print(f"Zero-byte files: {len(report.zero_byte_files)}")
    print(f"Top-level files under root: {len(report.top_level_files)}")
    print(f"Cases with missing/extra files: {len(report.bad_cases)}")
    print(f"Unreadable NIfTI files: {len(report.unreadable_files)}")
    print(f"Unexpected shapes: {len(report.unexpected_shapes)}")
    print(f"Unexpected segmentation label sets: {len(report.unexpected_seg_labels)}")
    print()

    def print_examples(title: str, values: list | dict, limit: int = 10) -> None:
        if not values:
            return
        print(title)
        items = values.items() if isinstance(values, dict) else enumerate(values)
        for index, value in list(items)[:limit]:
            print(f"  - {index}: {value}")
        if len(values) > limit:
            print(f"  ... and {len(values) - limit} more")
        print()

    print_examples("Top-level files:", [str(path) for path in report.top_level_files])
    print_examples("Zero-byte files:", [str(path) for path in report.zero_byte_files])
    print_examples("Bad cases:", report.bad_cases)
    print_examples("Unreadable files:", report.unreadable_files)
    print_examples("Unexpected shapes:", report.unexpected_shapes)
    print_examples("Unexpected segmentation labels:", report.unexpected_seg_labels)

    if report.ok:
        print("Verification PASSED")
    else:
        print("Verification FAILED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print BraTS 2021 download guidance or verify local dataset integrity."
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/BraTS2021",
        type=Path,
        help="Path to the BraTS2021 dataset root.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify local dataset structure and NIfTI readability.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only open a small sample of NIfTI files. Counts still cover all files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.verify:
        print_download_guidance(args.data_dir)
        return 0

    report = verify_dataset(args.data_dir, quick=args.quick)
    print_report(report, quick=args.quick)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
