"""Generate per-image predictions for the Dataset B CNN baseline.

This module converts Original Notebook Cell 124 into reusable code.
It records image metadata, true/predicted zero-based labels, grade names,
and per-class probabilities. Group-level metrics remain in ``evaluate.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_CLASS_NAMES = ("G1", "G2", "G3", "G4")
REQUIRED_BATCH_KEYS = {
    "image",
    "label",
    "subject_id",
    "group_id",
    "filename",
}


def _to_python_list(value, field_name: str) -> list:
    """Convert a collated metadata field into a normal Python list."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(
        f"Batch field {field_name!r} must be a tensor, list, or tuple."
    )


def generate_image_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run inference and optionally save image-level predictions as CSV.

    Parameters
    ----------
    model:
        Trained CNN model, normally restored with
        ``src.cnn.checkpoint.load_best_checkpoint``.
    loader:
        Test DataLoader whose batches contain image, label, subject_id,
        group_id, and filename fields.
    device:
        CPU or CUDA device used for model inference.
    class_names:
        Names corresponding to zero-based model output classes.
    output_path:
        Optional CSV destination. Parent directories are created when needed.

    Returns
    -------
    pandas.DataFrame
        One row per image with the source metadata, true/predicted class
        indices, grade names, and one probability column per class.
    """
    names = tuple(str(name) for name in class_names)
    if len(names) < 2:
        raise ValueError("class_names must contain at least two classes.")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique.")

    model = model.to(device)
    model.eval()

    filenames: list[str] = []
    subject_ids = []
    group_ids: list[str] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    all_probabilities: list[list[float]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if not isinstance(batch, dict):
                raise TypeError(
                    f"Batch {batch_index} must be a dictionary."
                )

            missing = REQUIRED_BATCH_KEYS.difference(batch)
            if missing:
                raise ValueError(
                    f"Batch {batch_index} is missing keys: "
                    + ", ".join(sorted(missing))
                )

            images = batch["image"]
            labels = batch["label"]

            if not torch.is_tensor(images) or not torch.is_tensor(labels):
                raise TypeError(
                    "Batch image and label fields must be PyTorch tensors."
                )

            images = images.to(device, non_blocking=True)
            labels_cpu = labels.detach().cpu().reshape(-1)

            logits = model(images)
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise ValueError(
                    "Model output must be a 2D logits tensor shaped (B, C)."
                )
            if logits.shape[0] != images.shape[0]:
                raise ValueError(
                    "Model output batch size does not match input batch size."
                )
            if logits.shape[1] != len(names):
                raise ValueError(
                    f"Model returned {logits.shape[1]} classes but "
                    f"{len(names)} class_names were supplied."
                )
            if labels_cpu.numel() != images.shape[0]:
                raise ValueError(
                    "Number of labels does not match the image batch size."
                )

            probabilities = torch.softmax(logits, dim=1)
            if not torch.isfinite(probabilities).all():
                raise ValueError(
                    f"Model produced non-finite probabilities in batch {batch_index}."
                )

            batch_predictions = probabilities.argmax(dim=1).cpu().tolist()
            batch_probabilities = probabilities.cpu().tolist()

            batch_filenames = _to_python_list(
                batch["filename"], "filename"
            )
            batch_subjects = _to_python_list(
                batch["subject_id"], "subject_id"
            )
            batch_groups = _to_python_list(
                batch["group_id"], "group_id"
            )

            batch_size = images.shape[0]
            if not (
                len(batch_filenames)
                == len(batch_subjects)
                == len(batch_groups)
                == batch_size
            ):
                raise ValueError(
                    f"Metadata length mismatch in batch {batch_index}."
                )

            filenames.extend(str(value) for value in batch_filenames)
            subject_ids.extend(batch_subjects)
            group_ids.extend(str(value) for value in batch_groups)
            true_labels.extend(int(value) for value in labels_cpu.tolist())
            predicted_labels.extend(
                int(value) for value in batch_predictions
            )
            all_probabilities.extend(
                [float(value) for value in row]
                for row in batch_probabilities
            )

    if not true_labels:
        raise ValueError("Prediction DataLoader produced no image records.")

    probability_array = np.asarray(all_probabilities, dtype=float)
    if probability_array.shape != (len(true_labels), len(names)):
        raise RuntimeError(
            "Collected probability array has an unexpected shape: "
            f"{probability_array.shape}."
        )

    if any(label < 0 or label >= len(names) for label in true_labels):
        raise ValueError("True labels fall outside the supplied class range.")

    prediction_df = pd.DataFrame(
        {
            "filename": filenames,
            "subject_id": subject_ids,
            "group_id": group_ids,
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "true_grade": [names[index] for index in true_labels],
            "predicted_grade": [names[index] for index in predicted_labels],
        }
    )

    for index, class_name in enumerate(names):
        prediction_df[f"{class_name}_probability"] = probability_array[:, index]

    if prediction_df["filename"].duplicated().any():
        duplicated = (
            prediction_df.loc[
                prediction_df["filename"].duplicated(keep=False),
                "filename",
            ]
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Image prediction output contains duplicate filenames: "
            f"{duplicated}"
        )

    if output_path is not None:
        csv_path = Path(output_path).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(csv_path, index=False)

    return prediction_df
