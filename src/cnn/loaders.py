"""DataLoader construction and batch checks for the Dataset B CNN baseline.

The defaults reproduce the original notebook's DataLoader settings:
batch size 32, two workers, pinned memory, shuffled training data, and
ordered validation/test data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .config import CONFIG


REQUIRED_BATCH_KEYS = {
    "image",
    "label",
    "subject_id",
    "group_id",
    "filename",
}


def create_data_loaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    *,
    batch_size: Optional[int] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train, validation, and test partitions."""
    resolved_batch_size = CONFIG.batch_size if batch_size is None else batch_size

    if not isinstance(resolved_batch_size, int) or resolved_batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }
    for name, dataset in datasets.items():
        if dataset is None:
            raise ValueError(f"{name}_dataset cannot be None.")
        if len(dataset) == 0:
            raise ValueError(f"{name} dataset is empty.")

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=resolved_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def validate_first_batch(
    loader: DataLoader,
    *,
    image_size: Optional[int] = None,
    num_classes: int = CONFIG.num_classes,
    expected_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Inspect one CPU batch and validate its shape, types, and labels."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")

    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("DataLoader produced no batches.") from exc

    if not isinstance(batch, dict):
        raise TypeError("Expected each dataset item to collate into a dictionary.")

    missing = REQUIRED_BATCH_KEYS.difference(batch)
    if missing:
        raise ValueError(
            "Batch is missing required keys: " + ", ".join(sorted(missing))
        )

    images = batch["image"]
    labels = batch["label"]

    if not torch.is_tensor(images) or not torch.is_tensor(labels):
        raise TypeError("'image' and 'label' batch values must be tensors.")
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"Expected image batch shape (B, 1, H, W), got {tuple(images.shape)}."
        )
    if image_size is not None and tuple(images.shape[-2:]) != (
        image_size,
        image_size,
    ):
        raise ValueError(
            f"Expected image dimensions {(image_size, image_size)}, "
            f"got {tuple(images.shape[-2:])}."
        )
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("Labels must have shape (B,) and match image batch size.")
    if expected_batch_size is not None and images.shape[0] != expected_batch_size:
        raise ValueError(
            f"Expected batch size {expected_batch_size}, got {images.shape[0]}."
        )
    if labels.dtype != torch.long:
        raise ValueError(f"Expected torch.long labels, got {labels.dtype}.")
    if not torch.is_floating_point(images):
        raise ValueError(f"Expected floating-point images, got {images.dtype}.")
    if not torch.isfinite(images).all():
        raise ValueError("Image batch contains NaN or infinite values.")
    if labels.numel() and (
        labels.min().item() < 0 or labels.max().item() >= num_classes
    ):
        raise ValueError(f"Labels must be in the range 0 to {num_classes - 1}.")

    return {
        "batch_size": int(images.shape[0]),
        "image_shape": tuple(images.shape),
        "image_dtype": str(images.dtype),
        "label_shape": tuple(labels.shape),
        "label_dtype": str(labels.dtype),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
        "image_device": str(images.device),
        "pinned_memory": bool(loader.pin_memory),
    }
