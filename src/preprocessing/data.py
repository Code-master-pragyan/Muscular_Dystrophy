"""
Prepare model-ready metadata for Dataset B FSHD ultrasound classification.

This module reuses the audited image manifest and the already approved
subject-level split. It never creates a new split and never combines Dataset A
(MRI) with Dataset B (ultrasound).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


VALID_PARTITIONS = ("train", "validation", "test")
VALID_GRADES = (1, 2, 3, 4)

REQUIRED_MANIFEST_COLUMNS = {
    "filename",
    "image_path",
    "subject_id",
    "muscle_code",
    "side_code",
    "heckmatt_grade",
}

REQUIRED_SPLIT_COLUMNS = {
    "subject_id",
    "partition",
}


def prepare_partition_dataframes(
    manifest_path: str | Path,
    split_path: str | Path,
) -> Dict[str, pd.DataFrame]:
    """
    Load the existing manifest and subject split and prepare model metadata.

    Parameters
    ----------
    manifest_path:
        Path to ``Dataset_B_Image_Mask_Manifest.csv``.
    split_path:
        Path to ``Dataset_B_Subject_Level_Split.csv``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        DataFrames keyed by ``train``, ``validation`` and ``test``.
        Each row remains one labelled image record. ``label`` is the
        zero-based class index (grade 1 -> 0, ..., grade 4 -> 3), and
        ``group_id`` identifies one subject–muscle–side group.

    Notes
    -----
    The split is joined by ``subject_id`` with a many-to-one validation.
    No random splitting, image transformation, mask processing, feature
    extraction, or dataset combination is performed here.
    """
    manifest_file = Path(manifest_path)
    split_file = Path(split_path)

    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Subject split file not found: {split_file}")

    manifest_df = pd.read_csv(manifest_file)
    split_df = pd.read_csv(split_file)

    missing_manifest = REQUIRED_MANIFEST_COLUMNS.difference(manifest_df.columns)
    missing_split = REQUIRED_SPLIT_COLUMNS.difference(split_df.columns)

    if missing_manifest:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_manifest))
        )
    if missing_split:
        raise ValueError(
            "Subject split is missing required columns: "
            + ", ".join(sorted(missing_split))
        )

    # The approved split must contain exactly one assignment per subject.
    if split_df["subject_id"].isna().any():
        raise ValueError("Subject split contains missing subject_id values.")
    if split_df["partition"].isna().any():
        raise ValueError("Subject split contains missing partition values.")
    if split_df["subject_id"].duplicated().any():
        duplicates = (
            split_df.loc[
                split_df["subject_id"].duplicated(keep=False), "subject_id"
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "Subject split must contain one row per subject. "
            f"Duplicate subject IDs include: {duplicates[:10]}"
        )

    split_df["partition"] = split_df["partition"].astype(str).str.strip()
    unexpected_partitions = (
        set(split_df["partition"].unique()) - set(VALID_PARTITIONS)
    )
    if unexpected_partitions:
        raise ValueError(
            f"Unexpected partition values: {sorted(unexpected_partitions)}. "
            f"Expected only: {list(VALID_PARTITIONS)}."
        )

    # Fail early when essential manifest metadata is absent.
    if manifest_df["subject_id"].isna().any():
        raise ValueError("Manifest contains missing subject_id values.")
    if manifest_df["filename"].isna().any():
        raise ValueError("Manifest contains missing filename values.")
    if manifest_df["image_path"].isna().any():
        raise ValueError("Manifest contains missing image_path values.")
    if manifest_df[["muscle_code", "side_code"]].isna().any().any():
        raise ValueError("Manifest contains missing muscle_code or side_code values.")

    # A partition column belongs to the split file. Avoid ambiguous merge suffixes.
    if "partition" in manifest_df.columns:
        raise ValueError(
            "Manifest already contains a 'partition' column. "
            "Pass the original audited manifest without an attached split."
        )

    # Retain only records with an available Heckmatt grade, as in the notebook.
    labelled_df = manifest_df.loc[
        manifest_df["heckmatt_grade"].notna()
    ].copy()

    if labelled_df.empty:
        raise ValueError("No labelled records were found in the manifest.")

    # Attach the previously approved subject-level split; do not resplit.
    labelled_df = labelled_df.merge(
        split_df[["subject_id", "partition"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = labelled_df["_merge"] != "both"
    if unmatched.any():
        missing_subjects = (
            labelled_df.loc[unmatched, "subject_id"].drop_duplicates().tolist()
        )
        raise ValueError(
            "Some labelled records have no approved subject partition. "
            f"Example subject IDs: {missing_subjects[:10]}"
        )

    labelled_df = labelled_df.drop(columns=["_merge"])

    grades = pd.to_numeric(labelled_df["heckmatt_grade"], errors="coerce")
    invalid_grades = grades.isna() | ~grades.isin(VALID_GRADES)
    if invalid_grades.any():
        invalid_values = (
            labelled_df.loc[invalid_grades, "heckmatt_grade"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Heckmatt grades must be integers in {1, 2, 3, 4}. "
            f"Invalid values include: {invalid_values[:10]}"
        )

    labelled_df["heckmatt_grade"] = grades.astype(int)
    labelled_df["label"] = labelled_df["heckmatt_grade"] - 1

    # Preserve the group definition used for group-level evaluation.
    labelled_df["group_id"] = (
        labelled_df["subject_id"].astype(str)
        + "_"
        + labelled_df["muscle_code"].astype(str)
        + "_"
        + labelled_df["side_code"].astype(str)
    )

    partitions = {
        name: (
            labelled_df.loc[labelled_df["partition"] == name]
            .copy()
            .reset_index(drop=True)
        )
        for name in VALID_PARTITIONS
    }

    # Ensure that no participant occurs in more than one partition.
    subject_sets = {
        name: set(frame["subject_id"].unique())
        for name, frame in partitions.items()
    }
    for index, first in enumerate(VALID_PARTITIONS):
        for second in VALID_PARTITIONS[index + 1 :]:
            overlap = subject_sets[first].intersection(subject_sets[second])
            if overlap:
                raise ValueError(
                    f"Subject leakage between {first} and {second}: "
                    f"{sorted(overlap)[:10]}"
                )

    # Every labelled record must appear in exactly one returned partition.
    partition_record_count = sum(len(frame) for frame in partitions.values())
    if partition_record_count != len(labelled_df):
        raise RuntimeError(
            "Partitioning lost or duplicated labelled image records: "
            f"{partition_record_count} returned versus {len(labelled_df)} labelled."
        )

    return partitions
