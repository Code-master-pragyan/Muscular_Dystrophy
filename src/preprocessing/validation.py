"""
Validation utilities for the FSHD ultrasound preprocessing pipeline.

This module validates the partition DataFrames returned by
src.preprocessing.data.prepare_partition_dataframes().

It does not create or modify manifests, subject splits, images, or masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd


EXPECTED_RECORD_COUNTS = {
    "train": 3342,
    "validation": 741,
    "test": 644,
}

REQUIRED_COLUMNS = {
    "filename",
    "image_path",
    "subject_id",
    "muscle_code",
    "side_code",
    "heckmatt_grade",
    "label",
    "group_id",
    "partition",
}


def validate_partition_dataframes(
    partitions: Mapping[str, pd.DataFrame],
    *,
    check_image_files: bool = True,
    check_mask_files: bool = False,
    expected_counts: Optional[Mapping[str, int]] = EXPECTED_RECORD_COUNTS,
) -> Dict[str, dict]:
    """
    Validate train/validation/test DataFrames.

    Args:
        partitions: Mapping with keys 'train', 'validation', and 'test'.
        check_image_files: If True, verify that every image_path exists.
        check_mask_files: If True, verify mask_path exists and every mask is present.
            Masks are not consumed by the current CNN Dataset implementation, so this
            check is disabled by default.
        expected_counts: Expected labelled record counts per partition. Pass None
            to skip exact count checks when intentionally working with a subset.

    Returns:
        A report dictionary containing record, subject, group, and class counts.

    Raises:
        ValueError: If a structural, label, leakage, or count check fails.
        FileNotFoundError: If a required image or mask file is missing.
    """
    required_partitions = {"train", "validation", "test"}
    if set(partitions.keys()) != required_partitions:
        raise ValueError(
            "partitions must contain exactly: train, validation, and test."
        )

    reports: Dict[str, dict] = {}
    subject_sets = {}
    all_filenames = []

    for partition_name in ("train", "validation", "test"):
        df = partitions[partition_name]

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{partition_name} must be a pandas DataFrame.")

        missing_columns = REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"{partition_name} is missing columns: {sorted(missing_columns)}"
            )

        if df.empty:
            raise ValueError(f"{partition_name} partition is empty.")

        if df[list(REQUIRED_COLUMNS)].isnull().any().any():
            null_columns = df[list(REQUIRED_COLUMNS)].columns[
                df[list(REQUIRED_COLUMNS)].isnull().any()
            ].tolist()
            raise ValueError(
                f"{partition_name} contains null values in: {null_columns}"
            )

        if not df["partition"].eq(partition_name).all():
