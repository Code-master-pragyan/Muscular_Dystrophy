"""Save validation and test evaluation artifacts for the CNN baseline.

This module converts the result-export block from the original CNN notebook
into a reusable function. It writes the same artifact types and filenames:
metrics CSV, test group predictions, validation/test confusion matrices,
classification reports, and experiment summary JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import CONFIG


REQUIRED_RESULT_KEYS = {
    "image_accuracy",
    "image_macro_f1",
    "group_accuracy",
    "group_macro_f1",
    "group_balanced_accuracy",
    "group_report",
    "group_confusion_matrix",
    "group_predictions",
}


def _validate_result(name: str, result: Mapping[str, Any]) -> None:
    """Check that an evaluation result has the keys used by the notebook."""
    missing = REQUIRED_RESULT_KEYS.difference(result)
    if missing:
        raise ValueError(
            f"{name} results are missing keys: {sorted(missing)}"
        )

    for metric in (
        "image_accuracy",
        "image_macro_f1",
        "group_accuracy",
        "group_macro_f1",
        "group_balanced_accuracy",
    ):
        try:
            value = float(result[metric])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} metric {metric!r} must be numeric."
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"{name} metric {metric!r} must be finite."
            )

    if not isinstance(result["group_predictions"], pd.DataFrame):
        raise TypeError(
            f"{name} group_predictions must be a pandas DataFrame."
        )

    matrix = np.asarray(result["group_confusion_matrix"])
    expected_shape = (CONFIG.num_classes, CONFIG.num_classes)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{name} confusion matrix must have shape {expected_shape}; "
            f"got {matrix.shape}."
        )


def save_cnn_evaluation_results(
    validation_results: Mapping[str, Any],
    test_results: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    output_dir: str | Path,
    *,
    class_names: Sequence[str] = ("G1", "G2", "G3", "G4"),
    random_seed: int = CONFIG.random_seed,
    maximum_epochs: int = CONFIG.max_epochs,
    early_stopping_patience: int = CONFIG.early_stopping_patience,
) -> dict[str, Path]:
    """Save the CNN validation/test outputs beneath ``evaluation_results``.

    Parameters
    ----------
    validation_results, test_results:
        Dictionaries returned by ``src.cnn.evaluate.evaluate_cnn``.
    checkpoint:
        Loaded checkpoint dictionary returned by
        ``src.cnn.checkpoint.load_best_checkpoint``.
    output_dir:
        CNN experiment directory. The function creates an
        ``evaluation_results`` subdirectory beneath it.
    class_names:
        Names used as confusion-matrix row and column labels. Defaults
        to the original notebook names G1–G4.
    random_seed, maximum_epochs, early_stopping_patience:
        Experiment settings recorded in the JSON summary.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths of all written result artifacts, keyed by artifact name.
    """
    _validate_result("Validation", validation_results)
    _validate_result("Test", test_results)

    if len(class_names) != CONFIG.num_classes:
        raise ValueError(
            f"class_names must contain {CONFIG.num_classes} entries."
        )

    required_checkpoint_keys = {
        "epoch",
        "best_val_group_macro_f1",
    }
    missing_checkpoint = required_checkpoint_keys.difference(checkpoint)
    if missing_checkpoint:
        raise ValueError(
            "Checkpoint is missing keys: "
            + ", ".join(sorted(missing_checkpoint))
        )

    best_epoch = int(checkpoint["epoch"])
    best_validation_score = float(
        checkpoint["best_val_group_macro_f1"]
    )
    if best_epoch < 1 or not np.isfinite(best_validation_score):
        raise ValueError("Checkpoint epoch/validation score is invalid.")

    base_dir = Path(output_dir).expanduser()
    if base_dir.exists() and not base_dir.is_dir():
        raise NotADirectoryError(
            f"CNN output path is not a directory: {base_dir}"
        )

    evaluation_dir = base_dir / "evaluation_results"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save validation and test image-/group-level metrics.
    metrics_summary = pd.DataFrame(
        [
            {
                "dataset": "validation",
                "image_accuracy": float(validation_results["image_accuracy"]),
                "image_macro_f1": float(validation_results["image_macro_f1"]),
                "group_accuracy": float(validation_results["group_accuracy"]),
                "group_macro_f1": float(validation_results["group_macro_f1"]),
                "group_balanced_accuracy": float(
                    validation_results["group_balanced_accuracy"]
                ),
            },
            {
                "dataset": "test",
                "image_accuracy": float(test_results["image_accuracy"]),
                "image_macro_f1": float(test_results["image_macro_f1"]),
                "group_accuracy": float(test_results["group_accuracy"]),
                "group_macro_f1": float(test_results["group_macro_f1"]),
                "group_balanced_accuracy": float(
                    test_results["group_balanced_accuracy"]
                ),
            },
        ]
    )
    metrics_path = evaluation_dir / "validation_test_metrics.csv"
    metrics_summary.to_csv(metrics_path, index=False)

    # 2. Save group-level test predictions.
    predictions_path = evaluation_dir / "test_group_predictions.csv"
    test_results["group_predictions"].to_csv(
        predictions_path,
        index=False,
    )

    # 3. Save group-level confusion matrices.
    validation_cm_path = evaluation_dir / "validation_confusion_matrix.csv"
    test_cm_path = evaluation_dir / "test_confusion_matrix.csv"

    pd.DataFrame(
        np.asarray(validation_results["group_confusion_matrix"]),
        index=list(class_names),
        columns=list(class_names),
    ).to_csv(validation_cm_path)

    pd.DataFrame(
        np.asarray(test_results["group_confusion_matrix"]),
        index=list(class_names),
        columns=list(class_names),
    ).to_csv(test_cm_path)

    # 4. Save classification reports as plain text.
    validation_report_path = (
        evaluation_dir / "validation_classification_report.txt"
    )
    test_report_path = (
        evaluation_dir / "test_classification_report.txt"
    )

    validation_report_path.write_text(
        str(validation_results["group_report"]),
        encoding="utf-8",
    )
    test_report_path.write_text(
        str(test_results["group_report"]),
        encoding="utf-8",
    )

    # 5. Save the experiment summary used by the original notebook.
    experiment_summary = {
        "model": "ResNet18",
        "dataset": "Dataset B - FSHD Ultrasound",
        "task": "Heckmatt grade classification",
        "classes": list(class_names),
        "best_epoch": best_epoch,
        "selection_metric": "Validation group-level macro-F1",
        "best_validation_group_macro_f1": best_validation_score,
        "test_group_macro_f1": float(test_results["group_macro_f1"]),
        "test_group_accuracy": float(test_results["group_accuracy"]),
        "test_group_balanced_accuracy": float(
            test_results["group_balanced_accuracy"]
        ),
        "test_image_accuracy": float(test_results["image_accuracy"]),
        "test_image_macro_f1": float(test_results["image_macro_f1"]),
        "random_seed": int(random_seed),
        "maximum_epochs": int(maximum_epochs),
        "early_stopping_patience": int(early_stopping_patience),
        "primary_evaluation": "Group-level mean probability aggregation",
        "secondary_evaluation": "Image-level classification",
    }
    summary_path = evaluation_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(experiment_summary, indent=4),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_path,
        "test_group_predictions": predictions_path,
        "validation_confusion_matrix": validation_cm_path,
        "test_confusion_matrix": test_cm_path,
        "validation_classification_report": validation_report_path,
        "test_classification_report": test_report_path,
        "experiment_summary": summary_path,
    }
