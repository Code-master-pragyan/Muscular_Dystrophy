"""Training utilities for the Dataset B CNN-radiomics hybrid XGBoost model.

Converts the original Colab hybrid training-matrix and XGBoost fit blocks
into reusable functions.

Important protocol:
- input rows are group-level records, not individual images;
- grades 1–4 are mapped to zero-based labels 0–3;
- balanced class weights are computed from training labels only;
- validation is used as the early-stopping evaluation set;
- the test partition is not passed to model.fit().
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


PARTITIONS = ("train", "validation", "test")
CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)
CNN_FEATURE_PREFIX = "cnn_feature_"
RADIOMICS_FEATURE_PREFIX = "original_"

DEFAULT_EXPECTED_GROUPS = {
    "train": 1029,
    "validation": 231,
    "test": 208,
}

DEFAULT_XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 4,
    "n_estimators": 1000,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 2.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": 2,
}


@dataclass(frozen=True)
class HybridTrainingMatrices:
    """Validated group-level matrices and training-only weighting metadata."""

    feature_columns: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_weights: dict[int, float]
    sample_weights: np.ndarray
    class_counts: dict[int, int]


def _feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Select hybrid features while preserving DataFrame column order."""
    selected = [
        str(column)
        for column in dataframe.columns
        if str(column).startswith(CNN_FEATURE_PREFIX)
        or str(column).startswith(RADIOMICS_FEATURE_PREFIX)
    ]
    if not selected:
        raise ValueError("No CNN or radiomics feature columns were found.")
    if len(selected) != len(set(selected)):
        raise ValueError("Feature column names must be unique.")
    return selected


def _validate_partition(
    dataframe: pd.DataFrame,
    partition: str,
    feature_columns: list[str],
    expected_groups: Optional[Mapping[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one group-level partition and return X and zero-based y."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{partition} must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError(f"{partition} partition is empty.")

    required = {"group_id", "heckmatt_grade", *feature_columns}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"{partition} is missing columns: {', '.join(sorted(missing))}"
        )

    if dataframe["group_id"].isna().any():
        raise ValueError(f"{partition} contains missing group_id values.")
    if dataframe["group_id"].astype(str).duplicated().any():
        raise ValueError(
            f"{partition} must contain exactly one row per group_id."
        )

    if "partition" in dataframe.columns:
        found = set(
            dataframe["partition"].astype(str).str.strip().str.lower()
        )
        if found != {partition}:
            raise ValueError(
                f"{partition} DataFrame contains partition values {sorted(found)}."
            )

    if expected_groups is not None and partition in expected_groups:
        expected = int(expected_groups[partition])
        if len(dataframe) != expected:
            raise ValueError(
                f"{partition} has {len(dataframe)} groups; expected {expected}."
            )

    grades = pd.to_numeric(dataframe["heckmatt_grade"], errors="coerce")
    if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
        raise ValueError(
            f"{partition} Heckmatt grades must be integers in {{1, 2, 3, 4}}."
        )
    if not (grades == grades.astype(int)).all():
        raise ValueError(f"{partition} Heckmatt grades must be integer values.")

    features = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(
            f"{partition} contains missing/non-numeric features: {bad[:10]}."
        )

    X = features.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"{partition} feature matrix contains non-finite values.")

    y = grades.astype(int).to_numpy(dtype=np.int64) - 1
    if y.shape != (len(dataframe),):
        raise RuntimeError(
            f"{partition} label vector does not match the feature row count."
        )

    return X, y


def prepare_hybrid_training_matrices(
    group_data: Mapping[str, pd.DataFrame],
    *,
    expected_groups: Optional[Mapping[str, int]] = DEFAULT_EXPECTED_GROUPS,
) -> HybridTrainingMatrices:
    """Prepare train/validation/test matrices and training-only class weights.

    The feature order is selected from the training DataFrame and must match
    exactly in the validation and test DataFrames. Class weights follow
    sklearn's ``class_weight='balanced'`` formula over classes 0, 1, 2, 3.
    """
    missing_partitions = set(PARTITIONS).difference(group_data)
    if missing_partitions:
        raise ValueError(
            "group_data must contain train, validation, and test; missing: "
            f"{sorted(missing_partitions)}."
        )

    feature_columns = _feature_columns(group_data["train"])

    matrices = {}
    labels = {}
    for partition in PARTITIONS:
        current_columns = _feature_columns(group_data[partition])
        if current_columns != feature_columns:
            raise ValueError(
                f"{partition} feature columns/order differ from training."
            )
        matrices[partition], labels[partition] = _validate_partition(
            group_data[partition],
            partition,
            feature_columns,
            expected_groups,
        )

    present_classes = np.unique(labels["train"])
    if not np.array_equal(present_classes, CLASSES):
        raise ValueError(
            "Training partition must contain all four classes 0–3 to compute "
            f"balanced weights. Found: {present_classes.tolist()}."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=CLASSES,
        y=labels["train"],
    )
    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(CLASSES, weights)
    }
    sample_weights = np.asarray(
        [class_weights[int(label)] for label in labels["train"]],
        dtype=np.float32,
    )

    if not np.isfinite(sample_weights).all() or not (sample_weights > 0).all():
        raise ValueError("Training sample weights must be finite and positive.")

    class_counts = {
        int(class_index): int((labels["train"] == class_index).sum())
        for class_index in CLASSES
    }

    return HybridTrainingMatrices(
        feature_columns=tuple(feature_columns),
        X_train=matrices["train"],
        y_train=labels["train"],
        X_validation=matrices["validation"],
        y_validation=labels["validation"],
        X_test=matrices["test"],
        y_test=labels["test"],
        class_weights=class_weights,
        sample_weights=sample_weights,
        class_counts=class_counts,
    )


def train_hybrid_xgboost(
    matrices: HybridTrainingMatrices,
    *,
    parameter_overrides: Optional[Mapping[str, object]] = None,
    verbose: int | bool = 50,
):
    """Fit the hybrid XGBClassifier with train-only sample weights.

    The notebook's exact hyperparameters are used unless explicitly
    overridden. The validation partition is passed as the early-stopping
    evaluation set. The test partition is deliberately not used.

    Returns
    -------
    (model, training_summary)
        Fitted XGBClassifier and a dictionary containing elapsed time,
        selected iteration, best validation log loss, boosting rounds,
        feature count, and the effective model parameters.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for hybrid model training. "
            "Install it with: pip install xgboost"
        ) from exc

    parameters = dict(DEFAULT_XGBOOST_PARAMS)
    if parameter_overrides:
        parameters.update(dict(parameter_overrides))

    # Keep the four-class task consistent with the notebook.
    if parameters.get("objective") != "multi:softprob":
        raise ValueError("Hybrid training requires objective='multi:softprob'.")
    if int(parameters.get("num_class", 4)) != 4:
        raise ValueError("Hybrid training requires num_class=4.")

    model = xgb.XGBClassifier(**parameters)

    started = perf_counter()
    model.fit(
        matrices.X_train,
        matrices.y_train,
        sample_weight=matrices.sample_weights,
        eval_set=[(matrices.X_validation, matrices.y_validation)],
        verbose=verbose,
    )
    elapsed_seconds = perf_counter() - started

    booster = model.get_booster()
    summary = {
        "xgboost_version": xgb.__version__,
        "training_time_seconds": float(elapsed_seconds),
        "training_time_minutes": float(elapsed_seconds / 60.0),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_validation_log_loss": float(
            getattr(model, "best_score", float("nan"))
        ),
        "total_boosting_rounds": int(booster.num_boosted_rounds()),
        "model_feature_count": int(model.n_features_in_),
        "training_groups": int(len(matrices.X_train)),
        "validation_groups": int(len(matrices.X_validation)),
        "test_groups_not_used_for_fit": int(len(matrices.X_test)),
        "class_counts_zero_based": matrices.class_counts,
        "class_weights_zero_based": matrices.class_weights,
        "parameters": parameters,
    }
    return model, summary
