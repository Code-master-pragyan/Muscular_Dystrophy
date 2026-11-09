"""Quality control helpers for PyRadiomics feature tables.

This module converts PyRadiomics scalar outputs (including zero-dimensional
NumPy arrays) into ordinary numeric values and audits the resulting feature
table for missing values, infinities, and zero-variance features.

The implementation is based on the feature inspection and cleanup steps in
Step_2_Dataset_Audit.ipynb (cells 84-87). It does not perform feature
selection, imputation, scaling, or model fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURE_PREFIX = "original_"


def get_radiomics_feature_columns(
    feature_table: pd.DataFrame,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
) -> list[str]:
    """Return the radiomics feature columns in their existing table order.

    Metadata columns such as ``filename``, ``subject_id``, ``group_id``,
    ``heckmatt_grade`` and ``partition`` are left untouched because they do not
    begin with the PyRadiomics feature prefix.
    """
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")
    if not feature_prefix:
        raise ValueError("feature_prefix must be a non-empty string.")

    columns = [
        column
        for column in feature_table.columns
        if isinstance(column, str) and column.startswith(feature_prefix)
    ]
    if not columns:
        raise ValueError(
            f"No feature columns beginning with {feature_prefix!r} were found."
        )
    return columns


def _to_numeric_scalar(value: Any) -> float:
    """Convert one scalar-like PyRadiomics value to float.

    PyRadiomics versions may return some features as NumPy scalar values and
    others as zero-dimensional or one-element NumPy arrays. Both are accepted.
    Multi-element arrays and non-numeric values become NaN, matching the
    notebook's cleanup behavior for non-scalar values.
    """
    if value is None:
        return np.nan

    try:
        array = np.asarray(value)
        if array.size != 1:
            return np.nan
        scalar = array.item()
        if scalar is None:
            return np.nan
        return float(scalar)
    except (TypeError, ValueError, OverflowError):
        return np.nan


def clean_radiomics_feature_table(
    feature_table: pd.DataFrame,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Convert selected PyRadiomics feature columns to numeric float columns.

    Values containing one scalar (including ``np.array(value)``) are converted
    to ``float``. Non-scalar or non-numeric values are replaced with ``NaN``.
    All non-feature metadata columns are preserved without modification.

    Args:
        feature_table: Table returned by the radiomics extraction pipeline.
        feature_prefix: Prefix identifying actual radiomics feature columns.
        copy: When true, leave the input table unchanged.

    Returns:
        A cleaned DataFrame with feature columns represented as float values.
    """
    feature_columns = get_radiomics_feature_columns(
        feature_table, feature_prefix=feature_prefix
    )
    cleaned = feature_table.copy(deep=True) if copy else feature_table

    for column in feature_columns:
        cleaned[column] = cleaned[column].map(_to_numeric_scalar).astype(float)

    return cleaned


def audit_radiomics_feature_table(
    feature_table: pd.DataFrame,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create per-feature and table-level quality summaries.

    The returned per-feature report contains:
      - feature name and dtype
      - missing and infinite value counts
      - number of distinct non-missing values
      - zero-variance flag

    Zero variance is assessed using the number of distinct non-missing values.
    This is a descriptive audit only; it does not remove columns.
    """
    feature_columns = get_radiomics_feature_columns(
        feature_table, feature_prefix=feature_prefix
    )

    numeric = feature_table[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    finite_mask = np.isfinite(values)

    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        series = numeric[column]
        column_values = series.to_numpy(dtype=float, na_value=np.nan)
        non_missing = series.dropna()
        unique_count = int(non_missing.nunique(dropna=True))
        rows.append(
            {
                "feature": column,
                "dtype": str(feature_table[column].dtype),
                "missing_values": int(series.isna().sum()),
                "infinite_values": int(np.isinf(column_values).sum()),
                "unique_values": unique_count,
                "zero_variance": bool(unique_count <= 1),
            }
        )

    report = pd.DataFrame(rows)
    summary = {
        "rows": int(len(feature_table)),
        "features": int(len(feature_columns)),
        "missing_feature_values": int(numeric.isna().sum().sum()),
        "infinite_feature_values": int(np.isinf(values).sum()),
        "zero_variance_features": int(report["zero_variance"].sum()),
        "non_finite_feature_values": int((~finite_mask).sum()),
    }
    return report, summary


def save_radiomics_quality_report(
    report: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a per-feature quality report as CSV and return its path."""
    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be a pandas DataFrame.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return path
