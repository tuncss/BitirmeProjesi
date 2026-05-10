"""Upload and result file management services."""

from __future__ import annotations

import shutil
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import nibabel as nib
from loguru import logger

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import InvalidFileFormatError, UploadedFileNotFoundError


REQUIRED_MODALITIES = ("t1", "t1ce", "t2", "flair")


class FileManager:
    """Manage uploaded NIfTI files, result directories, and cleanup."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.results_dir = Path(self.settings.results_dir)
        self.allowed_extensions = tuple(self.settings.allowed_extensions)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_upload_session(self) -> str:
        """Create a directory for one upload session and return its id."""
        session_id = uuid.uuid4().hex
        self.get_upload_session_dir(session_id, create=True)
        return session_id

    def get_upload_session_dir(self, session_id: str, create: bool = False) -> Path:
        """Return an upload session directory, optionally creating it."""
        _validate_storage_id(session_id, "session_id")
        session_dir = self.upload_dir / session_id
        if create:
            session_dir.mkdir(parents=True, exist_ok=True)
        elif not session_dir.exists():
            raise UploadedFileNotFoundError(session_id)
        return session_dir

    async def save_uploaded_file(self, session_id: str, filename: str, content: bytes) -> Path:
        """Validate and save one uploaded file into a session directory."""
        return self.save_file(session_id=session_id, filename=filename, content=content)

    def save_file(self, session_id: str, filename: str, content: bytes) -> Path:
        """Synchronous file save helper used by tests and API routes."""
        self.validate_extension(filename)
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise InvalidFileFormatError(filename)

        session_dir = self.get_upload_session_dir(session_id, create=True)
        target = session_dir / safe_name
        target.write_bytes(content)
        return target

    def extract_zip(self, session_id: str, zip_path: str | Path) -> list[Path]:
        """Extract a zip upload into its session directory with path traversal protection."""
        archive_path = Path(zip_path)
        self.validate_extension(archive_path.name)
        if not archive_path.exists():
            raise UploadedFileNotFoundError(str(archive_path))

        session_dir = self.get_upload_session_dir(session_id, create=True)
        extracted: list[Path] = []
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = Path(member.filename).name
                if not member_name:
                    continue
                self.validate_extension(member_name)

                target = _safe_join(session_dir, member_name)
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                extracted.append(target)

        return extracted

    def identify_modalities(self, session_id: str) -> dict[str, Path]:
        """Map uploaded filenames to required BraTS modalities."""
        session_dir = self.get_upload_session_dir(session_id)
        modality_map: dict[str, Path] = {}

        for path in sorted(self._iter_supported_files(session_dir)):
            modality = infer_modality_from_filename(path.name)
            if modality is None:
                continue
            if modality in modality_map:
                raise ValueError(f"Duplicate modality '{modality}' in upload session {session_id}")
            modality_map[modality] = path

        missing = sorted(set(REQUIRED_MODALITIES) - set(modality_map))
        if missing:
            raise ValueError(f"Missing modalities: {missing}")

        return {modality: modality_map[modality] for modality in REQUIRED_MODALITIES}

    def validate_nifti_files(self, modality_map: dict[str, str | Path]) -> bool:
        """Validate modality completeness, NIfTI readability, and shape consistency."""
        provided = set(modality_map)
        required = set(REQUIRED_MODALITIES)
        missing = sorted(required - provided)
        extra = sorted(provided - required)
        if missing:
            raise ValueError(f"Missing modalities: {missing}")
        if extra:
            raise ValueError(f"Unexpected modalities: {extra}")

        shapes: list[tuple[int, ...]] = []
        for modality in REQUIRED_MODALITIES:
            path = Path(modality_map[modality])
            if not path.exists():
                raise UploadedFileNotFoundError(str(path))
            try:
                image = nib.load(str(path))
                shapes.append(tuple(int(value) for value in image.shape[:3]))
            except Exception as exc:  # noqa: BLE001 - wrap nibabel errors with modality context
                raise ValueError(f"Invalid NIfTI file for {modality}: {exc}") from exc

        if len(set(shapes)) != 1:
            raise ValueError(f"NIfTI modality shapes must match, got: {shapes}")
        return True

    def create_result_dir(self, task_id: str) -> Path:
        """Create and return a result directory for a segmentation task."""
        _validate_storage_id(task_id, "task_id")
        result_dir = self.results_dir / task_id
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def get_result_path(self, task_id: str, filename: str = "segmentation.nii.gz") -> Path:
        """Return the expected path for a task output file."""
        _validate_storage_id(task_id, "task_id")
        return self.results_dir / task_id / Path(filename).name

    def cleanup_old_files(self, max_age_hours: int | None = None) -> list[Path]:
        """Remove upload/result session directories older than max_age_hours."""
        age_hours = max_age_hours if max_age_hours is not None else self.settings.cleanup_after_hours
        cutoff = datetime.now() - timedelta(hours=age_hours)
        removed: list[Path] = []

        for directory in (self.upload_dir, self.results_dir):
            if not directory.exists():
                continue
            for item in directory.iterdir():
                if not item.is_dir():
                    continue
                modified = datetime.fromtimestamp(item.stat().st_mtime)
                if modified < cutoff:
                    shutil.rmtree(item)
                    removed.append(item)
                    logger.info(f"Cleaned up old directory: {item}")

        return removed

    def validate_extension(self, filename: str) -> str:
        """Validate a filename extension and return the normalized extension."""
        extension = normalized_extension(filename)
        if extension not in self.allowed_extensions:
            raise InvalidFileFormatError(filename)
        return extension

    def _iter_supported_files(self, session_dir: Path) -> Iterable[Path]:
        for path in session_dir.rglob("*"):
            if path.is_file() and normalized_extension(path.name) in {".nii", ".nii.gz"}:
                yield path


def infer_modality_from_filename(filename: str) -> str | None:
    """Infer BraTS modality from a filename without confusing t1 and t1ce."""
    stem = _strip_medical_suffixes(Path(filename).name.lower())
    tokens = [token for token in stem.replace("-", "_").split("_") if token]

    for modality in ("t1ce", "flair", "t2", "t1"):
        if modality in tokens or stem.endswith(modality):
            return modality
    return None


def normalized_extension(filename: str) -> str:
    """Return .nii, .nii.gz, .zip, or the final suffix for other files."""
    lowered = filename.lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    if lowered.endswith(".nii"):
        return ".nii"
    if lowered.endswith(".zip"):
        return ".zip"
    return Path(lowered).suffix


def _strip_medical_suffixes(filename: str) -> str:
    for suffix in (".nii.gz", ".nii", ".zip"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def _validate_storage_id(value: str, field_name: str) -> None:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Invalid {field_name}: {value}")


def _safe_join(base_dir: Path, filename: str) -> Path:
    target = base_dir / Path(filename).name
    resolved_base = base_dir.resolve()
    resolved_target = target.resolve()
    if resolved_base not in resolved_target.parents and resolved_target != resolved_base:
        raise ValueError(f"Unsafe zip member path: {filename}")
    return target
