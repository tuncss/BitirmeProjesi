"""PyTorch Dataset and DataLoader utilities for preprocessed BraTS data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class BraTSDataset(Dataset):
    """BraTS 2021 dataset backed by compressed .npz preprocessing outputs."""

    def __init__(
        self,
        data_dir: str | Path,
        case_ids: list[str],
        augmentation=None,
        return_metadata: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.case_ids = list(case_ids)
        self.augmentation = augmentation
        self.return_metadata = return_metadata
        self._worker_seed: int | None = None

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        case_id = self.case_ids[idx]
        data_path = self.data_dir / f"{case_id}.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed case file not found: {data_path}")

        with np.load(data_path) as data:
            images = data["images"].astype(np.float32, copy=True)
            mask = data["mask"].astype(np.int64, copy=True)

        if self.augmentation is not None:
            self._maybe_seed_augmentation_worker()
            images, mask = self.augmentation(images, mask)

        images_tensor = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))

        if self.return_metadata:
            return images_tensor, mask_tensor, {"case_id": case_id}
        return images_tensor, mask_tensor

    def _maybe_seed_augmentation_worker(self) -> None:
        """Avoid duplicated augmentation RNG streams in multi-worker loaders."""
        if self.augmentation is None or not hasattr(self.augmentation, "rng"):
            return

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return

        seed = torch.initial_seed() % (2**32)
        if self._worker_seed == seed:
            return

        self.augmentation.rng = np.random.default_rng(seed)
        self._worker_seed = seed


def create_data_splits(
    data_dir: str | Path,
    config: dict[str, Any],
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split preprocessed case ids into train, validation, and test sets."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_path}")

    all_ids = sorted(path.stem for path in data_path.glob("*.npz"))
    if not all_ids:
        raise ValueError(f"No .npz files found in processed data directory: {data_path}")

    split_config = config["data"]["split"]
    train_ratio = float(split_config["train"])
    val_ratio = float(split_config["val"])
    test_ratio = float(split_config["test"])
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Train/val/test split ratios must sum to 1.0, got {ratio_sum}")

    train_ids, temp_ids = train_test_split(
        all_ids,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=relative_val,
        random_state=seed,
        shuffle=True,
    )

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def get_dataloaders(config: dict[str, Any], seed: int = 42) -> dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    from src.data.augmentation import BraTSAugmentation

    data_dir = Path(config["data"]["processed_dir"])
    batch_size = int(config["training"]["batch_size"])
    loader_config = config.get("data_loader", {})

    train_ids, val_ids, test_ids = create_data_splits(data_dir, config, seed=seed)

    train_dataset = BraTSDataset(
        data_dir=data_dir,
        case_ids=train_ids,
        augmentation=BraTSAugmentation(config),
    )
    val_dataset = BraTSDataset(
        data_dir=data_dir,
        case_ids=val_ids,
        augmentation=None,
    )
    test_dataset = BraTSDataset(
        data_dir=data_dir,
        case_ids=test_ids,
        augmentation=None,
        return_metadata=True,
    )

    train_workers = int(loader_config.get("train_num_workers", _default_num_workers(train=True)))
    eval_workers = int(loader_config.get("eval_num_workers", _default_num_workers(train=False)))
    pin_memory = bool(loader_config.get("pin_memory", torch.cuda.is_available()))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_seed_worker if train_workers > 0 else None,
        persistent_workers=train_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if eval_workers > 0 else None,
        persistent_workers=eval_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if eval_workers > 0 else None,
        persistent_workers=eval_workers > 0,
    )

    print(
        f"Train: {len(train_dataset)} cases, "
        f"Val: {len(val_dataset)} cases, "
        f"Test: {len(test_dataset)} cases"
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def _default_num_workers(train: bool) -> int:
    """Use safe local defaults on Windows and faster defaults elsewhere."""
    if os.name == "nt":
        return 0
    return 4 if train else 2


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy per DataLoader worker for reproducible stochastic transforms."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed((worker_seed + worker_id) % (2**32))
