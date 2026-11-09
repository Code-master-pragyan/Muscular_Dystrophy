"""Class-imbalance handling and optimizer setup for the CNN baseline.

This module reproduces Original Notebook Cell 107:
- class counts are calculated from the training partition only;
- inverse-frequency weights use N / (K * class_count);
- weighted CrossEntropyLoss is used;
- AdamW uses the project's configured learning rate and weight decay.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch
from torch import nn

from .config import CONFIG


def compute_training_class_weights(
    train_df: pd.DataFrame,
    device: torch.device,
    *,
    num_classes: int = CONFIG.num_classes,
) -> Tuple[pd.Series, torch.Tensor]:
    """Calculate inverse-frequency weights using training labels only.

    The input labels must already be zero-based integers from
    ``0`` to ``num_classes - 1``. Every class must be represented in
    the training partition, as required by the notebook's formula.
    """
    if "label" not in train_df.columns:
        raise ValueError("train_df must contain a 'label' column.")
    if train_df.empty:
        raise ValueError("train_df is empty; cannot calculate class weights.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    labels = pd.to_numeric(train_df["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError("Training labels contain missing or non-numeric values.")

    if not (labels == labels.astype(int)).all():
        raise ValueError("Training labels must be integer class indices.")

    labels = labels.astype(int)
    invalid = ~labels.isin(range(num_classes))
    if invalid.any():
        values = sorted(labels.loc[invalid].unique().tolist())
        raise ValueError(
            f"Training labels must be in 0..{num_classes - 1}; "
            f"found {values}."
        )

    class_counts = (
        labels.value_counts()
        .reindex(range(num_classes), fill_value=0)
        .sort_index()
    )

    missing_classes = class_counts[class_counts == 0].index.tolist()
    if missing_classes:
        raise ValueError(
            "Cannot compute finite inverse-frequency weights because "
            f"these classes are absent from the training partition: "
            f"{missing_classes}."
        )

    total_samples = int(class_counts.sum())
    weights = total_samples / (
        num_classes * torch.tensor(
            class_counts.to_numpy(),
            dtype=torch.float32,
        )
    )
    weights = weights.to(device)

    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Computed class weights must be finite and positive.")

    return class_counts, weights


def build_training_components(
    model: nn.Module,
    train_df: pd.DataFrame,
    device: torch.device,
    *,
    learning_rate: float = CONFIG.learning_rate,
    weight_decay: float = CONFIG.weight_decay,
    num_classes: int = CONFIG.num_classes,
):
    """Build weighted CrossEntropyLoss and AdamW from the training partition.

    Returns
    -------
    criterion, optimizer, class_counts, class_weights
        Objects ready to pass to ``src.cnn.train.train_cnn``.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    class_counts, class_weights = compute_training_class_weights(
        train_df,
        device,
        num_classes=num_classes,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return criterion, optimizer, class_counts, class_weights
