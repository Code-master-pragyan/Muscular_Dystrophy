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
