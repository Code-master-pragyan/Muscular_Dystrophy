
"""
Group-level aggregation utilities for radiomics feature tables.

The project evaluates predictions at the subject–muscle–side group level.
This module aggregates image-level radiomics features by group using the
arithmetic mean. It does not fit a model or select features.
"""

from __future__ import annotations

from typing import Optional, Sequence

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
    """

    if aggregation != "mean":
        raise ValueError("Only aggregation='mean' is currently supported.")

    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df must be a pandas DataFrame.")

    if features_df.empty:
        raise ValueError("features_df is empty.")

    group_columns = list(group_columns)

    if not group_columns:
        raise ValueError("At least one group column must be provided.")

    if len(group_columns) != len(set(group_columns)):
        raise ValueError("group_columns must not contain duplicates.")

    required_columns = set(group_columns) | {label_column}
    missing_columns = required_columns.difference(features_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    null_check_columns = group_columns + [label_column]

    if features_df[null_check_columns].isnull().any().any():
        null_columns = features_df[null_check_columns].columns[
            features_df[null_check_columns].isnull().any()
        ].tolist()

        raise ValueError(
            f"Null values in group or label columns: {null_columns}"
        )

    if feature_columns is None:
        excluded_columns = required_columns

        feature_columns = [
            column
            for column in features_df.select_dtypes(include="number").columns
            if column not in excluded_columns
        ]
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("No radiomics feature columns were selected.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns must not contain duplicates.")

    missing_features = set(feature_columns).difference(features_df.columns)

    if missing_features:
        raise ValueError(
            f"Selected feature columns are missing: {sorted(missing_features)}"
        )

    reserved_columns = required_columns | {"image_count"}
    conflicting_columns = reserved_columns.intersection(feature_columns)

    if conflicting_columns:
        raise ValueError(
            "Feature columns cannot overlap group, label, or reserved "
            f"output columns: {sorted(conflicting_columns)}"
        )

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(features_df[column])
    ]

    if non_numeric:
        raise ValueError(
            f"Radiomics features must be numeric: {non_numeric}"
        )

    grades = pd.to_numeric(
        features_df[label_column],
        errors="coerce"
    )

    if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
        raise ValueError(
            "Heckmatt grades must be integers from 1 through 4."
        )

    working = features_df[
        group_columns + [label_column] + list(feature_columns)
    ].copy()

    working[label_column] = grades.astype(int)

    # Every image in one group must have the same ground-truth grade.
    label_counts = (
        working.groupby(group_columns, dropna=False)[label_column]
        .nunique()
    )

    inconsistent_groups = label_counts[label_counts > 1]

    if not inconsistent_groups.empty:
        examples = list(inconsistent_groups.index[:5])

        raise ValueError(
            f"{len(inconsistent_groups)} group(s) contain inconsistent "
            f"Heckmatt grades. Examples: {examples}"
        )

    # Calculate mean radiomics features for each group.
    aggregation_spec = {
        label_column: (label_column, "first"),
        "image_count": (label_column, "size"),
    }

    aggregation_spec.update(
        {
            column: (column, "mean")
            for column in feature_columns
        }
    )

    result = (
        working.groupby(
            group_columns,
            as_index=False,
            sort=True,
            dropna=False,
        )
        .agg(**aggregation_spec)
    )

    expected_group_count = (
        working[group_columns].drop_duplicates().shape[0]
    )

    if len(result) != expected_group_count:
        raise RuntimeError(
            "Aggregation output group count does not match the input: "
            f"{len(result)} returned versus {expected_group_count} expected."
        )

    output_columns = (
        group_columns
        + [label_column, "image_count"]
        + list(feature_columns)
    )

    return result.loc[:, output_columns]
