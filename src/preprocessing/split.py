"""
Subject-level stratified split utilities for Dataset B FSHD ultrasound.

The original Colab workflow builds a subject-by-grade-presence matrix and
uses iterative multilabel stratification (random seed 42). It then validates
that all labelled subjects, groups, and images belong to exactly one
partition and saves the subject split plus a JSON design record.

Important provenance note:
The supplied notebook export did not retain the body of the cell that created
train_subjects, val_subjects, and test_subjects. Therefore, the candidate
split generator below implements the documented two-stage 70/15/15 method;
it must not be described as a byte-for-byte reconstruction of the missing
cell. The already-approved Dataset_B_Subject_Level_Split.csv remains the
canonical historical split and should not be overwritten casually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["subject_id", "muscle_code", "side_code"]
GRADE_PRESENCE_COLUMNS = [
    "grade_1_present",
    "grade_2_present",
    "grade_3_present",
    "grade_4_present",
]
PARTITION_ORDER = ("train", "validation", "test")
EXPECTED_SUBJECT_COUNTS = {
    "train": 76,
    "validation": 17,
    "test": 16,
}


def build_subject_grade_presence(
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build labelled groups and a subject-by-grade-presence matrix.

    This follows the original notebook cells that:
    1. retain rows with a Heckmatt grade,
    2. collapse images by subject, muscle, and side,
    3. create a binary subject-by-grade presence table.

    Returns
    -------
    labelled_images:
        All manifest image rows with a non-null Heckmatt grade.
    labelled_groups:
        One row per subject–muscle–side group with its grade and image count.
    subject_grade_presence:
        One row per subject and four binary grade-presence columns.
    """
    required = set(GROUP_COLUMNS + ["heckmatt_grade", "filename"])
    missing = required.difference(manifest_df.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    labelled_images = manifest_df.loc[
        manifest_df["heckmatt_grade"].notna()
    ].copy()

    if labelled_images.empty:
        raise ValueError("Manifest contains no labelled Heckmatt records.")

    grades = pd.to_numeric(
        labelled_images["heckmatt_grade"], errors="coerce"
    )
    if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
        raise ValueError("Heckmatt grades must be values 1, 2, 3, or 4.")

    labelled_images["heckmatt_grade"] = grades.astype(int)

    # A subject–muscle–side group must have one consistent Heckmatt grade.
    group_grade_counts = (
        labelled_images.groupby(GROUP_COLUMNS)["heckmatt_grade"].nunique()
    )
    inconsistent = group_grade_counts[group_grade_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Some subject–muscle–side groups contain inconsistent grades. "
            f"Examples: {list(inconsistent.index[:5])}"
        )

    labelled_groups = (
        labelled_images.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            grade=("heckmatt_grade", "first"),
            image_count=("filename", "count"),
        )
    )

    subject_grade_presence = pd.crosstab(
        labelled_groups["subject_id"],
        labelled_groups["grade"],
    ).gt(0).astype(int)

    subject_grade_presence = subject_grade_presence.reindex(
        columns=[1, 2, 3, 4],
        fill_value=0,
    )
    subject_grade_presence.columns = GRADE_PRESENCE_COLUMNS

    if subject_grade_presence.index.has_duplicates:
        raise ValueError("Subject-by-grade matrix contains duplicate subject IDs.")

    if not set(np.unique(subject_grade_presence.to_numpy())).issubset({0, 1}):
        raise ValueError("Grade-presence matrix must contain only 0 and 1.")

    return labelled_images, labelled_groups, subject_grade_presence


def generate_candidate_subject_split(
    subject_grade_presence: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a candidate 3-way subject split using iterative multilabel stratification.

    A two-stage split is used:
    - First, separate the training subjects from the remaining subjects.
    - Then, divide the remaining subjects into validation and test partitions.

    This implements the documented iterative multilabel method and 70/15/15
    target proportions. The missing original notebook cell means this function
    cannot guarantee reproduction of the historical subject assignments.

    Requires the ``iterative-stratification`` package.
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "Install the split dependency with: "
            "pip install iterative-stratification"
        ) from exc

    if not isinstance(subject_grade_presence, pd.DataFrame):
        raise TypeError("subject_grade_presence must be a pandas DataFrame.")

    if subject_grade_presence.empty:
        raise ValueError("subject_grade_presence is empty.")

    missing_columns = set(GRADE_PRESENCE_COLUMNS).difference(
        subject_grade_presence.columns
    )
    if missing_columns:
        raise ValueError(
            "Grade-presence matrix is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if subject_grade_presence.index.has_duplicates:
        raise ValueError("Subject IDs in the presence matrix must be unique.")

    if subject_grade_presence.index.isna().any():
        raise ValueError("Subject IDs in the presence matrix cannot be missing.")

    y = subject_grade_presence[GRADE_PRESENCE_COLUMNS].to_numpy(dtype=int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Grade-presence matrix must contain only binary values.")

    if len(subject_grade_presence) < 4:
        raise ValueError("At least four subjects are required for a 3-way split.")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction <= 0:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    subject_ids = subject_grade_presence.index.to_numpy()
    all_indices = np.arange(len(subject_ids))

    # Stage 1: training versus the temporary validation/test pool.
    first_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - train_fraction,
        random_state=random_seed,
    )
    train_indices, temporary_indices = next(
        first_split.split(np.zeros(len(y)), y)
    )

    temporary_y = y[temporary_indices]

    # Stage 2: validation versus test, preserving the requested overall ratios.
    validation_share_of_temporary = (
        validation_fraction / (validation_fraction + test_fraction)
    )
    second_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - validation_share_of_temporary,
        random_state=random_seed,
    )
    validation_local_indices, test_local_indices = next(
        second_split.split(
            np.zeros(len(temporary_y)),
            temporary_y,
        )
    )

    validation_indices = temporary_indices[validation_local_indices]
    test_indices = temporary_indices[test_local_indices]

    # Defensive integrity checks before materializing the split table.
    partition_indices = {
        "train": set(train_indices.tolist()),
        "validation": set(validation_indices.tolist()),
        "test": set(test_indices.tolist()),
    }

    for i, first in enumerate(PARTITION_ORDER):
        for second in PARTITION_ORDER[i + 1:]:
            if partition_indices[first].intersection(partition_indices[second]):
                raise RuntimeError(
                    f"Index overlap detected between {first} and {second}."
                )

    assigned_indices = set().union(*partition_indices.values())
    if assigned_indices != set(all_indices.tolist()):
        raise RuntimeError("Candidate split did not assign every subject exactly once.")

    partition_by_subject = {}
    for partition_name, indices in partition_indices.items():
        for index in indices:
            partition_by_subject[subject_ids[index]] = partition_name

    split_df = pd.DataFrame(
        {
            "subject_id": subject_ids.astype(str),
            "partition": [
                partition_by_subject[subject_id]
                for subject_id in subject_ids
            ],
        }
    )

    validate_subject_split(
        split_df,
        expected_subject_ids=subject_ids.astype(str),
    )
    return split_df


def validate_subject_split(
    split_df: pd.DataFrame,
    *,
    expected_subject_ids: Optional[Iterable[str]] = None,
    expected_counts: Optional[dict[str, int]] = None,
) -> dict:
    """
    Validate subject-level split assignments and return partition statistics.
    """
    required_columns = {"subject_id", "partition"}
    missing = required_columns.difference(split_df.columns)
    if missing:
        raise ValueError(
            f"Split DataFrame is missing columns: {sorted(missing)}"
        )

    if split_df.empty:
        raise ValueError("Subject split is empty.")

    split = split_df[["subject_id", "partition"]].copy()
    if split.isna().any().any():
        raise ValueError("Subject split contains missing values.")

    split["subject_id"] = split["subject_id"].astype(str)
    split["partition"] = split["partition"].astype(str).str.strip()

    if split["subject_id"].duplicated().any():
        duplicates = (
            split.loc[
                split["subject_id"].duplicated(keep=False), "subject_id"
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Each subject must have exactly one partition. "
            f"Duplicate IDs: {duplicates[:10]}"
        )

    unexpected = set(split["partition"].unique()) - set(PARTITION_ORDER)
    if unexpected:
        raise ValueError(f"Unexpected partition values: {sorted(unexpected)}")

    if set(split["partition"].unique()) != set(PARTITION_ORDER):
        raise ValueError("All three partitions must be represented.")

    if expected_subject_ids is not None:
        expected_ids = set(map(str, expected_subject_ids))
        actual_ids = set(split["subject_id"])
        if actual_ids != expected_ids:
            missing_ids = sorted(expected_ids - actual_ids)
            extra_ids = sorted(actual_ids - expected_ids)
            raise ValueError(
                "Split subject IDs do not match the expected subject set. "
                f"Missing: {missing_ids[:10]}; extra: {extra_ids[:10]}"
            )

    counts = {
        name: int((split["partition"] == name).sum())
        for name in PARTITION_ORDER
    }

    if expected_counts is not None and counts != expected_counts:
        raise ValueError(
            f"Partition counts are {counts}; expected {expected_counts}."
        )

    return {
        "total_subjects": int(len(split)),
        "partition_counts": counts,
        "subject_overlap": 0,
        "status": "Integrity checks passed",
    }


def save_subject_split_artifacts(
    split_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    labelled_groups: Optional[int] = None,
    labelled_images: Optional[int] = None,
    random_seed: int = 42,
) -> tuple[Path, Path]:
    """
    Save the subject split CSV and JSON design record, as in the original cells.

    The function writes only to the explicitly supplied output directory.
    It does not overwrite an existing split unless overwrite=True is passed.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    validation_report = validate_subject_split(split_df)

    split_path = output_root / "Dataset_B_Subject_Level_Split.csv"
    info_path = output_root / "Dataset_B_Split_Design_Record.json"

    if split_path.exists() or info_path.exists():
        raise FileExistsError(
            "Split artifact already exists. Move it or choose a new output "
            "directory to avoid overwriting the approved historical split."
        )

    split_df[["subject_id", "partition"]].to_csv(split_path, index=False)

    counts = validation_report["partition_counts"]
    split_info = {
        "dataset": "Dataset B - FSHD Ultrasound",
        "split_method": "Iterative multilabel stratification",
        "stratification_unit": "Subject",
        "stratification_targets": GRADE_PRESENCE_COLUMNS,
        "random_seed": int(random_seed),
        "train_subjects": counts["train"],
        "validation_subjects": counts["validation"],
        "test_subjects": counts["test"],
        "total_subjects": validation_report["total_subjects"],
        "labelled_groups": labelled_groups,
        "labelled_images": labelled_images,
        "subject_overlap": 0,
        "status": validation_report["status"],
        "implementation_note": (
            "Candidate split generated by the repository utility. "
            "The exact original notebook split-generation cell was absent "
            "from the supplied notebook export; compare with the approved "
            "historical CSV before treating these assignments as equivalent."
        ),
    }

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2)

    return split_path, info_path
