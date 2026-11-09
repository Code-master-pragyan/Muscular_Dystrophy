"""Checkpoint restoration for the Dataset B ResNet18 CNN baseline.

This module converts Original Notebook Cell 113 into a reusable function.
It expects the checkpoint format written by ``src.cnn.train.train_cnn``:
model_state_dict, optimizer_state_dict, epoch, and
best_val_group_macro_f1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn


REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "epoch",
    "best_val_group_macro_f1",
}


def load_best_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Restore the best validation-selected weights and set evaluation mode.

    Parameters
    ----------
    model:
        Model instance with the same architecture used during training.
    checkpoint_path:
        Path to ``resnet18_best_model.pth`` created by ``train_cnn``.
    device:
        Target device, for example ``torch.device("cuda")`` or
        ``torch.device("cpu")``.
    strict:
        Passed to ``model.load_state_dict``. Defaults to True so missing or
        unexpected parameter keys are not silently ignored.

    Returns
    -------
    (model, checkpoint)
        The model moved to ``device`` and set to evaluation mode, plus the
        loaded checkpoint dictionary containing the epoch and validation
        selection score.
    """
    checkpoint_file = Path(checkpoint_path).expanduser()

    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {checkpoint_file}"
        )

    # Keep the source notebook's explicit safe weights-only loading behavior.
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary.")

    missing_keys = REQUIRED_CHECKPOINT_KEYS.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )

    if not isinstance(checkpoint["model_state_dict"], dict):
        raise TypeError("checkpoint['model_state_dict'] must be a dictionary.")

    try:
        epoch = int(checkpoint["epoch"])
        selection_score = float(checkpoint["best_val_group_macro_f1"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint epoch and best validation score must be numeric."
        ) from exc

    if epoch < 1:
        raise ValueError(f"Checkpoint epoch must be >= 1; got {epoch}.")
    if not torch.isfinite(torch.tensor(selection_score)):
        raise ValueError("Checkpoint validation score must be finite.")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )
    model = model.to(device)
    model.eval()

    return model, checkpoint
