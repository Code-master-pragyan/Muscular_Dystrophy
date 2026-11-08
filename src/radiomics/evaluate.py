"""
Evaluation utilities for Dataset B FSHD ultrasound radiomics predictions.

This module implements the evaluation and group-aggregation logic used to
recalculate metrics from saved image-level prediction tables.

Important:
- Heckmatt grades are represented as 1, 2, 3, and 4 in the input/output.
- Probability columns are expected in the order G1, G2, G3, G4.
- Group predictions are produced by averaging image probabilities within each
  group_id, then taking the class with the highest mean probability.
- This file does not train or reproduce the original saved XGBoost model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


DEFAULT_PROBABILITY_COLUMNS = (
    "probability_G1",
    "probability_G2",
    "probability_G3",
    "probability_G4",
)
DEFAULT_CLASSES = (1, 2, 3, 4)


def _validate_prediction_columns(
    predictions: pd.DataFrame,
    probability_columns: Sequence[str],
    label_column: str,
) -> None:
    """Raise a clear error if required prediction columns are missing."""
    required = [label_column, *probability_columns]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    if len(probability_columns) != 4:
        raise ValueError("Exactly four probability columns are required for grades 1–4.")

    if predictions.empty:
        raise ValueError("The prediction table is empty.")


def aggregate_image_predictions_to_groups(
    image_predictions: pd.DataFrame,
    group_column: str = "group_id",
    label_column: str = "heckmatt_grade",
    probability_columns: Sequence[str] = DEFAULT_PROBABILITY_COLUMNS,
) -> pd.DataFrame:
    """
    Aggregate image-level probabilities into subject–muscle–side group predictions.

    Mean probability is calculated separately for each grade within each group.
    The group label is taken from the image rows after verifying that each group
    contains only one ground-truth grade.
    """
    if group_column not in image_predictions.columns:
        raise ValueError(f"Missing group column: {group_column}")

    _validate_prediction_columns(
        image_predictions, probability_columns, label_column
    )

    if image_predictions[group_column].isna().any():
        raise ValueError(f"{group_column} contains missing values.")

    probabilities = image_predictions.loc[:, probability_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if probabilities.isna().any().any():
        raise ValueError("Probability columns contain missing or non-numeric values.")

    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError("Probability columns contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError("Probabilities must be between 0 and 1.")

    work = image_predictions[[group_column, label_column]].copy()
    work.loc[:, list(probability_columns)] = probabilities

    label_counts = work.groupby(group_column, dropna=False)[label_column].nunique()
    inconsistent_groups = label_counts[label_counts != 1]
    if not inconsistent_groups.empty:
        raise ValueError(
            "Each group must have exactly one ground-truth grade. "
            f"Inconsistent groups: {inconsistent_groups.index.tolist()[:10]}"
        )

    grouped = (
        work.groupby(group_column, sort=True)
        .agg(
            **{label_column: (label_column, "first")},
            **{
                column: (column, "mean")
                for column in probability_columns
            },
        )
        .reset_index()
    )

    probability_matrix = grouped.loc[:, probability_columns].to_numpy()
    predicted_indices = np.argmax(probability_matrix, axis=1)
    grouped["predicted_grade"] = np.asarray(DEFAULT_CLASSES)[predicted_indices]

    return grouped


def calculate_classification_metrics(
    y_true,
    y_pred,
    labels: Sequence[int] = DEFAULT_CLASSES,
) -> dict:
    """Calculate accuracy, macro-F1, balanced accuracy, and class-wise report."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        raise ValueError("Cannot calculate metrics for an empty target array.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of items.")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(labels)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(labels),
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate_image_predictions(
    image_predictions: pd.DataFrame,
    label_column: str = "heckmatt_grade",
    prediction_column: str = "predicted_grade",
) -> dict:
    """Evaluate saved image-level predictions."""
    if label_column not in image_predictions or prediction_column not in image_predictions:
        raise ValueError(
            f"Prediction table must contain '{label_column}' and "
            f"'{prediction_column}'."
        )

    return calculate_classification_metrics(
        image_predictions[label_column],
        image_predictions[prediction_column],
    )


def evaluate_group_predictions(
    image_predictions: pd.DataFrame,
    group_column: str = "group_id",
    label_column: str = "heckmatt_grade",
    probability_columns: Sequence[str] = DEFAULT_PROBABILITY_COLUMNS,
) -> tuple[pd.DataFrame, dict]:
    """
    Aggregate image predictions to groups and calculate group-level metrics.

    Returns:
        (group_predictions, metrics)
    """
    group_predictions = aggregate_image_predictions_to_groups(
        image_predictions=image_predictions,
        group_column=group_column,
        label_column=label_column,
        probability_columns=probability_columns,
    )
    metrics = calculate_classification_metrics(
        group_predictions[label_column],
        group_predictions["predicted_grade"],
    )
    return group_predictions, metrics
