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

PARTITION_NAMES = ("train", "validation", "test")


def validate_partition_dataframes(
    partitions: Mapping[str, pd.DataFrame],
    *,
    check_image_files: bool = True,
    check_mask_files: bool = False,
    expected_counts: Optional[Mapping[str, int]] = EXPECTED_RECORD_COUNTS,
) -> Dict[str, dict]:
    """
    Validate train, validation, and test DataFrames.

    Parameters
    ----------
    partitions:
        Mapping with exactly these keys: train, validation, test.
    check_image_files:
        If True, verify that each image_path points to an existing file.
    check_mask_files:
        If True, require mask_path and verify that each mask exists.
        This is optional because the CNN Dataset does not consume masks.
    expected_counts:
        Expected number of labelled image records per partition. Pass None
        when validating a deliberately reduced or synthetic dataset.

    Returns
    -------
    dict
        A report with record, subject, group, and class counts for each
        partition, plus combined totals.

    Raises
    ------
    TypeError
        If a partition value is not a pandas DataFrame.
    ValueError
        If required columns, labels, group definitions, split integrity, or
        expected record counts are invalid.
    FileNotFoundError
        If a required image or mask file does not exist.
    """
    required_partitions = set(PARTITION_NAMES)
    if set(partitions.keys()) != required_partitions:
        raise ValueError(
            "partitions must contain exactly: train, validation, and test."
        )

    reports: Dict[str, dict] = {}
    subject_sets = {}
    all_filenames = []
    all_groups = set()

    for partition_name in PARTITION_NAMES:
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

        required_values = df[list(REQUIRED_COLUMNS)]
        if required_values.isnull().any().any():
            null_columns = required_values.columns[
                required_values.isnull().any()
            ].tolist()
            raise ValueError(
                f"{partition_name} contains null values in: {null_columns}"
            )

        if not df["partition"].astype(str).eq(partition_name).all():
            raise ValueError(
                f"Rows in the {partition_name} DataFrame have an incorrect "
                "partition value."
            )

        grades = pd.to_numeric(df["heckmatt_grade"], errors="coerce")
        labels = pd.to_numeric(df["label"], errors="coerce")

        if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
            raise ValueError(
                f"{partition_name} contains invalid Heckmatt grades; expected 1–4."
            )

        if labels.isna().any() or not labels.isin([0, 1, 2, 3]).all():
            raise ValueError(
                f"{partition_name} contains invalid labels; expected 0–3."
            )

        if not (
            labels.astype(int).to_numpy()
            == grades.astype(int).to_numpy() - 1
        ).all():
            raise ValueError(
                f"{partition_name}: label must equal Heckmatt grade minus 1."
            )

        if df["filename"].astype(str).duplicated().any():
            duplicates = (
                df.loc[
                    df["filename"].astype(str).duplicated(keep=False),
                    "filename",
                ]
                .astype(str)
                .head(5)
                .tolist()
            )
            raise ValueError(
                f"{partition_name} contains duplicate filenames, "
                f"for example: {duplicates}"
            )

        # Reconstruct the known project group key to catch accidental mismatch.
        expected_group_ids = (
            df["subject_id"].astype(str)
            + "_"
            + df["muscle_code"].astype(str)
            + "_"
            + df["side_code"].astype(str)
        )
        if not df["group_id"].astype(str).eq(expected_group_ids).all():
            raise ValueError(
                f"{partition_name} contains group_id values inconsistent "
                "with subject_id, muscle_code, and side_code."
            )

        if check_image_files:
            missing_images = [
                str(path)
                for path in df["image_path"]
                if not Path(str(path)).is_file()
            ]
            if missing_images:
                raise FileNotFoundError(
                    f"{partition_name}: {len(missing_images)} image file(s) "
                    f"not found. Examples: {missing_images[:5]}"
                )

        if check_mask_files:
            if "mask_path" not in df.columns:
                raise ValueError(
                    "check_mask_files=True, but the DataFrame has no mask_path column."
                )
            missing_masks = [
                str(path)
                for path in df["mask_path"]
                if pd.isna(path) or not Path(str(path)).is_file()
            ]
            if missing_masks:
                raise FileNotFoundError(
                    f"{partition_name}: {len(missing_masks)} mask file(s) "
                    f"not found. Examples: {missing_masks[:5]}"
                )

        subject_sets[partition_name] = set(df["subject_id"].astype(str))
        all_filenames.extend(df["filename"].astype(str).tolist())
        all_groups.update(df["group_id"].astype(str).unique().tolist())

        if expected_counts is not None and partition_name in expected_counts:
            expected = int(expected_counts[partition_name])
            actual = len(df)
            if actual != expected:
                raise ValueError(
                    f"{partition_name} has {actual} records; expected {expected}. "
                    "Pass expected_counts=None when intentionally validating "
                    "a subset."
                )

        reports[partition_name] = {
            "records": int(len(df)),
            "subjects": int(df["subject_id"].nunique()),
            "groups": int(df["group_id"].nunique()),
            "class_counts": {
                int(label): int(count)
                for label, count in (
                    df["label"].astype(int).value_counts().sort_index().items()
                )
            },
        }

    # Check subject-level isolation across all partitions.
    for index, first in enumerate(PARTITION_NAMES):
        for second in PARTITION_NAMES[index + 1 :]:
            overlap = subject_sets[first].intersection(subject_sets[second])
            if overlap:
                raise ValueError(
                    f"Subject leakage between {first} and {second}: "
                    f"{sorted(overlap)[:5]}"
                )

    # Filenames should identify unique image records across the whole split.
    if len(all_filenames) != len(set(all_filenames)):
        raise ValueError("Duplicate filenames found across partitions.")

    total_records = sum(reports[name]["records"] for name in PARTITION_NAMES)
    total_subjects = len(set().union(*subject_sets.values()))

    reports["total"] = {
        "records": int(total_records),
        "subjects": int(total_subjects),
        "groups": int(len(all_groups)),
    }

    if expected_counts is not None and all(
        name in expected_counts for name in PARTITION_NAMES
    ):
        expected_total = sum(
            int(expected_counts[name]) for name in PARTITION_NAMES
        )
        if total_records != expected_total:
            raise ValueError(
                f"Total records are {total_records}; expected {expected_total}."
            )

    return reports
