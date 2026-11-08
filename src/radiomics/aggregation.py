"""
Group-level aggregation utilities for radiomics feature tables.

The project evaluates predictions at the subject–muscle–side group level.
This module aggregates image-level radiomics features by group using the
arithmetic mean. It does not fit a model or select features.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


DEFAULT_GROUP_COLUMNS = ("subject_id", "muscle_code", "side_code")


def aggregate_radiomics_by_group(
    features_df: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    label_column: str = "heckmatt_grade",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate image-level radiomics features into subject–muscle–side groups.

    Args:
        features_df: Image-level feature table with group identifiers,
            Heckmatt grade, and numeric radiomics feature columns.
        feature_columns: Explicit radiomics feature columns to aggregate.
            If None, all numeric columns except identifiers and the label
            are selected.
        group_columns: Columns defining one subject–muscle–side group.
        label_column: Ground-truth Heckmatt grade column (1–4).
        aggregation: Currently only arithmetic mean is supported.

    Returns:
        One row per group, containing group identifiers, the group's label,
        image_count, and mean feature values.

    Raises:
        ValueError: If required columns are missing, labels disagree within a
            group, feature columns are invalid, or the input is empty.
    """
    if aggregation != "mean":
        raise ValueError("Only aggregation='mean' is currently supported.")

    if features_df.empty:
        raise ValueError("features_df is empty.")

    required = set(group_columns) | {label_column}
    missing = required.difference(features_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if features_df[list(required)].isnull().any().any():
        null_columns = features_df[list(required)].columns[
            features_df[list(required)].isnull().any()
        ].tolist()
        raise ValueError(f"Null values in group or label columns: {null_columns}")

    if feature_columns is None:
        excluded = set(group_columns) | {label_column}
        feature_columns = [
            column
            for column in features_df.select_dtypes(include="number").columns
            if column not in excluded
        ]
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("No radiomics feature columns were selected.")

    missing_features = set(feature_columns).difference(features_df.columns)
    if missing_features:
        raise ValueError(
            f"Selected feature columns are missing: {sorted(missing_features)}"
        )

    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(features_df[column])
    ]
    if non_numeric:
        raise ValueError(f"Radiomics features must be numeric: {non_numeric}")

    grades = pd.to_numeric(features_df[label_column], errors="coerce")
    if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
        raise ValueError("Heckmatt grades must be integers from 1 through 4.")

    working = features_df.copy()
    working[label_column] = grades.astype(int)
