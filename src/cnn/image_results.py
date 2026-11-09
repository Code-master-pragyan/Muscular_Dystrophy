"""Independent verification and export of CNN image-level test results.

Converted from Original Notebook Cells 125–126. Metrics are recalculated
from the image-level prediction table rather than copied from the training
or group-level evaluation output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


DEFAULT_CLASS_NAMES = ("G1", "G2", "G3", "G4")

# Reference values recorded in the original notebook for comparison only.
ORIGINAL_RECORDED_ACCURACY = 0.723602
ORIGINAL_RECORDED_MACRO_F1 = 0.686826

REQUIRED_COLUMNS = {
    "filename",
    "subject_id",
    "group_id",
    "true_label",
    "predicted_label",
}


def verify_and_save_image_level_results(
    image_predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    expected_images: int | None = 644,
    expected_groups: int | None = 208,
    reference_accuracy: float | None = ORIGINAL_RECORDED_ACCURACY,
    reference_macro_f1: float | None = ORIGINAL_RECORDED_MACRO_F1,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Recalculate image-level metrics and save the notebook's verification files.

    Parameters
    ----------
    image_predictions:
        DataFrame returned by ``src.cnn.predictions.generate_image_predictions``.
        It must contain one row per image with ground-truth and predicted labels.
    output_dir:
        The CNN ``evaluation_results`` directory. It is created if absent.
    class_names:
        Ordered labels for zero-based classes 0..K-1.
    expected_images:
        Expected number of test image records from the approved split.
        Defaults to the notebook's 644 test records. Pass None to disable.
    expected_groups:
        Expected number of subject–muscle–side groups in the test partition.
        Defaults to the notebook's 208 groups. Pass None to disable.
    reference_accuracy, reference_macro_f1:
        Metrics recorded in the original notebook for independent comparison.
        Pass None to skip the corresponding reference comparison.
    tolerance:
        Absolute tolerance used when comparing recalculated metrics with
        the recorded reference values.

    Returns
    -------
    dict
        Recalculated metrics, classification report, confusion matrix,
        reference comparisons, and paths of the saved artifacts.
    """
    missing = REQUIRED_COLUMNS.difference(image_predictions.columns)
    if missing:
        raise ValueError(
            "Image prediction table is missing columns: "
            + ", ".join(sorted(missing))
        )
    if image_predictions.empty:
        raise ValueError("Image prediction table is empty.")

    names = tuple(str(name) for name in class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique names.")

    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number.")

    predictions = image_predictions.copy()

    if predictions["filename"].isna().any():
        raise ValueError("Image prediction table contains missing filenames.")
    if predictions["filename"].astype(str).duplicated().any():
        raise ValueError("Image prediction table contains duplicate filenames.")
    if predictions[["subject_id", "group_id"]].isna().any().any():
        raise ValueError("Image prediction table has missing subject/group IDs.")

    for column in ("true_label", "predicted_label"):
        converted = pd.to_numeric(predictions[column], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"{column} contains missing/non-numeric values.")
        if not (converted == converted.astype(int)).all():
            raise ValueError(f"{column} must contain integer class indices.")
        if not converted.astype(int).between(0, len(names) - 1).all():
            raise ValueError(
                f"{column} values must be in 0..{len(names) - 1}."
            )
        predictions[column] = converted.astype(int)

    image_count = int(len(predictions))
    group_count = int(predictions["group_id"].nunique())
    subject_count = int(predictions["subject_id"].nunique())

    if expected_images is not None and image_count != expected_images:
        raise ValueError(
            f"Expected {expected_images} test image records, found {image_count}."
        )
    if expected_groups is not None and group_count != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} test groups, found {group_count}."
        )

    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["predicted_label"].to_numpy(dtype=int)
    class_indices = list(range(len(names)))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
    )
    balanced_accuracy = float(
        balanced_accuracy_score(y_true, y_pred)
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_indices,
        target_names=list(names),
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_indices,
    )

    accuracy_matches = (
        None
        if reference_accuracy is None
        else bool(np.isclose(accuracy, reference_accuracy, atol=tolerance))
    )
    macro_f1_matches = (
        None
        if reference_macro_f1 is None
        else bool(np.isclose(macro_f1, reference_macro_f1, atol=tolerance))
    )

    comparisons = [
        value for value in (accuracy_matches, macro_f1_matches)
        if value is not None
    ]
    if not comparisons:
        verification_status = "Recalculated; no reference comparison requested"
    elif all(comparisons):
        verification_status = (
            "Matched original recorded accuracy and macro-F1"
        )
    else:
        verification_status = (
            "One or more metrics did not match original recorded values"
        )

    verified_metrics = {
        "evaluation_level": "Image-level",
        "dataset": "Dataset B - FSHD Ultrasound",
        "partition": "Test",
        "number_of_images": image_count,
        "number_of_subjects": subject_count,
        "number_of_groups": group_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "reference_accuracy": reference_accuracy,
        "reference_macro_f1": reference_macro_f1,
        "accuracy_matches_reference": accuracy_matches,
        "macro_f1_matches_reference": macro_f1_matches,
        "comparison_tolerance": float(tolerance),
        "verification_status": verification_status,
    }

    evaluation_dir = Path(output_dir).expanduser()
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Save the predictions alongside the independently recalculated outputs.
    predictions_path = evaluation_dir / "test_image_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    confusion_path = evaluation_dir / "test_image_confusion_matrix.csv"
    pd.DataFrame(
        matrix,
        index=list(names),
        columns=list(names),
    ).to_csv(confusion_path)

    report_path = evaluation_dir / "test_image_classification_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    metrics_path = evaluation_dir / "test_image_verified_metrics.json"
    metrics_path.write_text(
        json.dumps(verified_metrics, indent=4),
        encoding="utf-8",
    )

    return {
        "metrics": verified_metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "reference_comparisons": {
            "accuracy_matches": accuracy_matches,
            "macro_f1_matches": macro_f1_matches,
        },
        "saved_files": {
            "predictions": predictions_path,
            "confusion_matrix": confusion_path,
            "classification_report": report_path,
            "verified_metrics": metrics_path,
        },
    }
