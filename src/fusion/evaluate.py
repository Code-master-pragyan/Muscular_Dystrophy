"""Evaluation and export for the Dataset B hybrid XGBoost model.

Converts the original Colab validation/test evaluation and test artifact
export blocks into reusable functions.

The source notebook explicitly used ``iteration_range=(0, 13)`` for the
final held-out test predictions, even though the saved booster contained
more rounds. Therefore callers must provide ``selected_rounds`` explicitly.
For the saved notebook test outputs, pass ``selected_rounds=13``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

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
METADATA_COLUMNS = (
    "group_id",
    "subject_id",
    "muscle_code",
    "side_code",
    "heckmatt_grade",
)
CNN_FEATURE_PREFIX = "cnn_feature_"
RADIOMICS_FEATURE_PREFIX = "original_"


def get_hybrid_feature_columns(
    group_df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    """Return hybrid feature columns, preserving their table order."""
    if feature_columns is not None:
        columns = list(feature_columns)
        missing = set(columns).difference(group_df.columns)
        if missing:
            raise ValueError(
                "Requested feature columns are missing: "
                + ", ".join(sorted(missing))
            )
        if not columns:
            raise ValueError("feature_columns cannot be empty.")
        if len(columns) != len(set(columns)):
            raise ValueError("feature_columns contains duplicate names.")
        return columns

    columns = [
        str(column)
        for column in group_df.columns
        if str(column).startswith(CNN_FEATURE_PREFIX)
        or str(column).startswith(RADIOMICS_FEATURE_PREFIX)
    ]
    if not columns:
        raise ValueError("No CNN/radiomics feature columns were found.")
    return columns


def evaluate_hybrid_partition(
    model,
    group_df: pd.DataFrame,
    *,
    partition: str,
    selected_rounds: int,
    feature_columns: Optional[Sequence[str]] = None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> dict:
    """Evaluate one group-level partition using an explicit number of rounds.

    Input labels are Heckmatt grades 1–4. They are converted to zero-based
    labels for XGBoost prediction and converted back to grades in the
    returned group prediction table.

    The model is expected to be an XGBoost classifier supporting
    ``predict`` and ``predict_proba`` with the ``iteration_range`` argument.
    """
    partition = str(partition).strip().lower()
    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test.")
    if not isinstance(selected_rounds, int) or selected_rounds < 1:
        raise ValueError("selected_rounds must be a positive integer.")

    names = tuple(str(name) for name in class_names)
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("Exactly four unique class_names are required.")

    if not isinstance(group_df, pd.DataFrame) or group_df.empty:
        raise ValueError("group_df must be a non-empty pandas DataFrame.")

    required = set(METADATA_COLUMNS)
    missing = required.difference(group_df.columns)
    if missing:
        raise ValueError(
            "Group table is missing metadata columns: "
            + ", ".join(sorted(missing))
        )
    if group_df[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Group table contains missing group metadata.")
    if group_df["group_id"].astype(str).duplicated().any():
        raise ValueError("Group table must contain one row per group_id.")

    if "partition" in group_df.columns:
        actual_partitions = set(
            group_df["partition"].astype(str).str.strip().str.lower()
        )
        if actual_partitions != {partition}:
            raise ValueError(
                f"Expected only {partition!r} rows; "
                f"found {sorted(actual_partitions)}."
            )

    columns = get_hybrid_feature_columns(group_df, feature_columns)
    X = group_df[columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Hybrid feature matrix contains missing/non-numeric values.")
    X_array = X.to_numpy(dtype=np.float32)
    if not np.isfinite(X_array).all():
        raise ValueError("Hybrid feature matrix contains non-finite values.")

    grades = pd.to_numeric(group_df["heckmatt_grade"], errors="coerce")
    if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
        raise ValueError("Heckmatt grades must be integers from 1 through 4.")
    if not (grades == grades.astype(int)).all():
        raise ValueError("Heckmatt grades must be integer values.")
    y_true = grades.astype(int).to_numpy(dtype=np.int64) - 1

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("model must provide predict() and predict_proba().")

    if hasattr(model, "get_booster"):
        trained_rounds = int(model.get_booster().num_boosted_rounds())
        if selected_rounds > trained_rounds:
            raise ValueError(
                f"selected_rounds={selected_rounds} exceeds the model's "
                f"{trained_rounds} available boosting rounds."
            )

    iteration_range = (0, selected_rounds)
    y_pred = np.asarray(
        model.predict(X_array, iteration_range=iteration_range),
        dtype=np.int64,
    ).reshape(-1)
    probabilities = np.asarray(
        model.predict_proba(X_array, iteration_range=iteration_range),
        dtype=np.float64,
    )

    if y_pred.shape != y_true.shape:
        raise ValueError("Predicted labels do not match the target row count.")
    if not np.isin(y_pred, [0, 1, 2, 3]).all():
        raise ValueError("Model predictions contain labels outside 0–3.")
    if probabilities.shape != (len(group_df), 4):
        raise ValueError(
            f"Expected probability shape {(len(group_df), 4)}, "
            f"received {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Predicted probabilities must be within [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Predicted class probabilities do not sum to one.")

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
    )
    balanced_accuracy = float(
        balanced_accuracy_score(y_true, y_pred)
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=list(names),
        output_dict=True,
        zero_division=0,
    )

    predictions = group_df[
        [
            "group_id",
            "subject_id",
            "muscle_code",
            "side_code",
            "heckmatt_grade",
        ]
    ].copy()
    predictions["predicted_grade"] = y_pred + 1

    for index in range(4):
        predictions[f"probability_G{index + 1}"] = probabilities[:, index]

    metrics = {
        "partition": partition,
        "groups": int(len(y_true)),
        "selected_boosting_rounds": int(selected_rounds),
        "feature_count": int(len(columns)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "classification_report": report,
    }

    return {
        "metrics": metrics,
        "confusion_matrix": matrix,
        "classification_report": report,
        "predictions": predictions,
        "probabilities": probabilities,
        "y_true": y_true,
        "y_pred": y_pred,
        "feature_columns": columns,
    }


def save_hybrid_evaluation_results(
    evaluation: dict,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save metrics JSON, confusion matrix CSV, and group prediction CSV.

    The test partition uses the original notebook filenames:
    ``Hybrid_Test_Metrics.json``,
    ``Hybrid_Test_Confusion_Matrix.csv``, and
    ``Hybrid_Test_Group_Predictions.csv``.
    """
    if not isinstance(evaluation, dict) or "metrics" not in evaluation:
        raise ValueError("evaluation must be returned by evaluate_hybrid_partition().")

    metrics = evaluation["metrics"]
    partition = str(metrics.get("partition", "")).strip().lower()
    if partition not in {"train", "validation", "test"}:
        raise ValueError("Evaluation result has an invalid partition.")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    partition_title = partition.capitalize()
    metrics_path = output_path / f"Hybrid_{partition_title}_Metrics.json"
    confusion_path = (
        output_path / f"Hybrid_{partition_title}_Confusion_Matrix.csv"
    )
    predictions_path = (
        output_path / f"Hybrid_{partition_title}_Group_Predictions.csv"
    )

    targets = [metrics_path, confusion_path, predictions_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation files: "
            + ", ".join(str(path) for path in existing)
            + ". Pass overwrite=True to replace them."
        )

    # Convert sklearn/numpy values to ordinary JSON-compatible Python values.
    def to_builtin(value):
        if isinstance(value, dict):
            return {str(key): to_builtin(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_builtin(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    metrics_path.write_text(
        json.dumps(to_builtin(metrics), indent=4),
        encoding="utf-8",
    )

    matrix = np.asarray(evaluation["confusion_matrix"])
    if matrix.shape != (4, 4):
        raise ValueError("Confusion matrix must have shape (4, 4).")
    pd.DataFrame(
        matrix,
        index=["Actual_G1", "Actual_G2", "Actual_G3", "Actual_G4"],
        columns=["Pred_G1", "Pred_G2", "Pred_G3", "Pred_G4"],
    ).to_csv(confusion_path)

    predictions = evaluation["predictions"]
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("evaluation predictions must be a DataFrame.")
    predictions.to_csv(predictions_path, index=False)

    return {
        "metrics": metrics_path,
        "confusion_matrix": confusion_path,
        "group_predictions": predictions_path,
    }
