"""Data augmentation utilities for BraTS 3D segmentation volumes."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, rotate


class BraTSAugmentation:
    """3D augmentation pipeline for BraTS images and segmentation masks.

    Geometric transforms are applied to images and mask with the same sampled
    parameters. Intensity transforms are applied only to image channels.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int | None = None):
        root_config = config or {}
        self.config = root_config.get("augmentation", root_config)
        self.enabled = bool(self.config.get("enabled", True))
        self.rng = np.random.default_rng(seed)

        rotation_config = self.config.get("random_rotation", {})
        elastic_config = self.config.get("elastic_deformation", {})

        self.flip_enabled = bool(self.config.get("random_flip", True))
        self.rotation_enabled = bool(rotation_config.get("enabled", True))
        self.elastic_enabled = bool(elastic_config.get("enabled", True))

        self.max_rotation_angle = float(rotation_config.get("max_angle", 15))
        self.elastic_alpha = float(elastic_config.get("alpha", 100))
        self.elastic_sigma = float(elastic_config.get("sigma", 10))

        self.flip_probability = float(self.config.get("flip_probability", 0.5))
        self.rotation_probability = float(self.config.get("rotation_probability", 0.5))
        self.intensity_probability = float(self.config.get("intensity_probability", 0.5))
        self.noise_probability = float(self.config.get("noise_probability", 0.3))
        self.elastic_probability = float(self.config.get("elastic_probability", 0.2))

        self.intensity_shift_range = float(self.config.get("intensity_shift_range", 0.1))
        self.noise_sigma = float(self.config.get("noise_sigma", 0.01))

    def __call__(self, images: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations.

        Args:
            images: Array with shape (4, D, H, W).
            mask: Array with shape (D, H, W).

        Returns:
            Augmented images and mask with the same shapes.
        """
        self._validate_inputs(images, mask)

        if not self.enabled:
            return images, mask

        augmented_images = images.astype(np.float32, copy=True)
        augmented_mask = mask.copy()

        if self.flip_enabled:
            for axis in range(3):
                augmented_images, augmented_mask = self.random_flip(
                    augmented_images,
                    augmented_mask,
                    axis=axis,
                    p=self.flip_probability,
                )

        if self.rotation_enabled and self.rng.random() < self.rotation_probability:
            augmented_images, augmented_mask = self.random_rotation_3d(
                augmented_images,
                augmented_mask,
                max_angle=self.max_rotation_angle,
            )

        if self.rng.random() < self.intensity_probability:
            augmented_images = self.random_intensity_shift(
                augmented_images,
                shift_range=self.intensity_shift_range,
            )

        if self.rng.random() < self.noise_probability:
            augmented_images = self.random_gaussian_noise(
                augmented_images,
                sigma=self.noise_sigma,
            )

        if self.elastic_enabled and self.rng.random() < self.elastic_probability:
            augmented_images, augmented_mask = self.elastic_deformation_3d(
                augmented_images,
                augmented_mask,
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma,
            )

        augmented_mask = np.rint(augmented_mask).astype(mask.dtype, copy=False)
        self._validate_outputs(images, mask, augmented_images, augmented_mask)
        return augmented_images.astype(np.float32, copy=False), augmented_mask

    def random_flip(
        self,
        images: np.ndarray,
        mask: np.ndarray,
        axis: int,
        p: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly flip along a spatial axis.

        axis is 0=depth, 1=height, 2=width for the mask. Images have one extra
        leading channel dimension, so the flip axis is shifted by one.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

        if self.rng.random() < p:
            images = np.flip(images, axis=axis + 1).copy()
            mask = np.flip(mask, axis=axis).copy()
        return images, mask

    def random_rotation_3d(
        self,
        images: np.ndarray,
        mask: np.ndarray,
        max_angle: float = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a random 3D rotation around one random plane."""
        angle = float(self.rng.uniform(-max_angle, max_angle))
        axes_pairs = ((0, 1), (0, 2), (1, 2))
        axes = axes_pairs[int(self.rng.integers(len(axes_pairs)))]

        rotated_images = np.zeros_like(images, dtype=np.float32)
        for channel in range(images.shape[0]):
            rotated_images[channel] = rotate(
                images[channel],
                angle,
                axes=axes,
                reshape=False,
                order=3,
                mode="constant",
                cval=0,
            )

        rotated_mask = rotate(
            mask,
            angle,
            axes=axes,
            reshape=False,
            order=0,
            mode="constant",
            cval=0,
        ).astype(mask.dtype, copy=False)

        return rotated_images, rotated_mask

    def random_intensity_shift(
        self,
        images: np.ndarray,
        shift_range: float = 0.1,
    ) -> np.ndarray:
        """Apply random scale and shift to non-zero voxels per modality."""
        shifted = images.astype(np.float32, copy=True)
        for channel in range(shifted.shape[0]):
            non_zero = shifted[channel] != 0
            if not non_zero.any():
                continue
            shift = float(self.rng.uniform(-shift_range, shift_range))
            scale = float(self.rng.uniform(1 - shift_range, 1 + shift_range))
            shifted[channel][non_zero] = shifted[channel][non_zero] * scale + shift
        return shifted

    def random_gaussian_noise(self, images: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to image voxels while preserving zero background."""
        noise = self.rng.normal(0, sigma, size=images.shape).astype(np.float32)
        noisy = images.astype(np.float32, copy=False) + noise
        noisy[images == 0] = 0
        return noisy.astype(np.float32, copy=False)

    def elastic_deformation_3d(
        self,
        images: np.ndarray,
        mask: np.ndarray,
        alpha: float | None = None,
        sigma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply 3D elastic deformation with a shared displacement field."""
        alpha = self.elastic_alpha if alpha is None else float(alpha)
        sigma = self.elastic_sigma if sigma is None else float(sigma)
        shape = images.shape[1:]

        dx = gaussian_filter(self.rng.standard_normal(shape), sigma, mode="reflect") * alpha
        dy = gaussian_filter(self.rng.standard_normal(shape), sigma, mode="reflect") * alpha
        dz = gaussian_filter(self.rng.standard_normal(shape), sigma, mode="reflect") * alpha

        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        indices = [z + dz, y + dy, x + dx]

        deformed_images = np.zeros_like(images, dtype=np.float32)
        for channel in range(images.shape[0]):
            deformed_images[channel] = map_coordinates(
                images[channel],
                indices,
                order=3,
                mode="constant",
                cval=0,
            )

        deformed_mask = map_coordinates(
            mask,
            indices,
            order=0,
            mode="constant",
            cval=0,
        ).astype(mask.dtype, copy=False)

        return deformed_images, deformed_mask

    @staticmethod
    def _validate_inputs(images: np.ndarray, mask: np.ndarray) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must have shape (C, D, H, W), got {images.shape}")
        if mask.ndim != 3:
            raise ValueError(f"mask must have shape (D, H, W), got {mask.shape}")
        if images.shape[1:] != mask.shape:
            raise ValueError(f"image/mask shape mismatch: {images.shape[1:]} vs {mask.shape}")

    @staticmethod
    def _validate_outputs(
        original_images: np.ndarray,
        original_mask: np.ndarray,
        augmented_images: np.ndarray,
        augmented_mask: np.ndarray,
    ) -> None:
        if augmented_images.shape != original_images.shape:
            raise ValueError(
                f"Augmented image shape changed: {original_images.shape} -> {augmented_images.shape}"
            )
        if augmented_mask.shape != original_mask.shape:
            raise ValueError(
                f"Augmented mask shape changed: {original_mask.shape} -> {augmented_mask.shape}"
            )
        allowed_labels = set(np.unique(original_mask).astype(int).tolist()) | {0}
        output_labels = set(np.unique(augmented_mask).astype(int).tolist())
        if not output_labels.issubset(allowed_labels):
            raise ValueError(f"Mask labels changed unexpectedly: {sorted(output_labels)}")

