"""Image-level and subject–muscle–side group-level CNN evaluation."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import CONFIG


def evaluate_cnn(model, loader, device):
    """Evaluate a CNN and aggregate image probabilities within each group.

    Each group must contain images with one consistent ground-truth label.
    Group prediction is the argmax of the mean class probabilities across
    images in that group, matching the original notebook's protocol.

    Returns a dictionary containing image-level metrics and group-level
    metrics, report, confusion matrix, and group predictions.
    """
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_subjects = []
    all_groups = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())
            all_probabilities.extend(probabilities.tolist())

            subject_ids = batch["subject_id"]
            if torch.is_tensor(subject_ids):
                subject_ids = subject_ids.cpu().numpy().tolist()
            all_subjects.extend(list(subject_ids))

            group_ids = batch["group_id"]
            all_groups.extend(list(group_ids))

    if not all_labels:
        raise ValueError("The evaluation DataLoader produced no samples.")

    image_labels = np.asarray(all_labels, dtype=int)
    image_predictions = np.asarray(all_predictions, dtype=int)
    image_probabilities = np.asarray(all_probabilities, dtype=float)

    labels = list(range(CONFIG.num_classes))
    class_names = list(CONFIG.class_names)

    prediction_df = pd.DataFrame(
        {
            "subject_id": all_subjects,
            "group_id": all_groups,
            "true_label": image_labels,
        }
    )

    for class_index in labels:
        prediction_df[f"prob_{class_index}"] = image_probabilities[:, class_index]

    group_records = []
    probability_columns = [f"prob_{i}" for i in labels]

    for group_id, group in prediction_df.groupby("group_id", sort=False):
        if group["true_label"].nunique() != 1:
            raise ValueError(f"Conflicting ground-truth labels in group {group_id}")

        mean_probabilities = group[probability_columns].mean().to_numpy()
        group_records.append(
            {
                "group_id": group_id,
                "true_label": int(group["true_label"].iloc[0]),
                "predicted_label": int(np.argmax(mean_probabilities)),
            }
        )

    group_df = pd.DataFrame(group_records)
    group_labels = group_df["true_label"].to_numpy(dtype=int)
    group_predictions = group_df["predicted_label"].to_numpy(dtype=int)

    return {
        "image_accuracy": accuracy_score(image_labels, image_predictions),
        "image_macro_f1": f1_score(
            image_labels,
            image_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "group_accuracy": accuracy_score(group_labels, group_predictions),
        "group_macro_f1": f1_score(
            group_labels,
            group_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "group_balanced_accuracy": balanced_accuracy_score(
            group_labels, group_predictions
        ),
        "group_report": classification_report(
            group_labels,
            group_predictions,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        ),
        "group_confusion_matrix": confusion_matrix(
            group_labels, group_predictions, labels=labels
        ),
        "group_predictions": group_df,
    }
