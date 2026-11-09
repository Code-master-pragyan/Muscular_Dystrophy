"""
Build the Dataset B FSHD ultrasound image-mask manifest.

This module extracts subject, muscle, side, and image-index metadata from
the original ultrasound filenames and attaches Heckmatt grades from
SubjectsInfo.xlsx. It follows the manifest-building logic in the original
Colab dataset-audit notebook.

Dataset A (MRI) is not processed by this module.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d{2})_(\d+)\.png$")

MANIFEST_COLUMNS = [
    "filename",
    "image_path",
    "mask_path",
    "mask_exists",
    "subject_id",
    "muscle_code",
    "side_code",
    "side",
    "image_index",
    "heckmatt_grade",
    "metadata_available",
]


def build_image_mask_manifest(
    image_dir: str | Path,
    mask_dir: str | Path,
    metadata_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build a manifest for Dataset B ultrasound images.

    Parameters
    ----------
    image_dir:
        Directory containing the original ultrasound PNG images.
    mask_dir:
        Directory containing masks with matching filenames.
    metadata_path:
        Path to ``SubjectsInfo.xlsx``.
    output_path:
        Optional CSV path. If supplied, the manifest is saved there.

    Returns
    -------
    pandas.DataFrame
        One row per image with the original manifest fields:
        filename, image_path, mask_path, mask_exists, subject_id,
        muscle_code, side_code, side, image_index, heckmatt_grade,
        and metadata_available.

    Notes
    -----
    The filename convention is:
        subject_id_muscle_code_side_code_image_index.png

    Heckmatt metadata is read from the column named
    ``{muscle_code:03d}_{side_code:02d}``, indexed by the integer subject Code.
    Files that do not match the filename pattern are skipped, as in the
    original notebook.
    """
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    metadata_file = Path(metadata_path)

    if not image_root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_root}")
    if not mask_root.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_root}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Subject metadata file not found: {metadata_file}")

    metadata = pd.read_excel(metadata_file)

    if "Code" not in metadata.columns:
        raise ValueError("SubjectsInfo.xlsx must contain a 'Code' column.")

    metadata = metadata.copy()
    metadata["Code"] = pd.to_numeric(
        metadata["Code"], errors="raise"
    ).astype(int)

    if metadata["Code"].duplicated().any():
        duplicate_ids = (
            metadata.loc[metadata["Code"].duplicated(keep=False), "Code"]
            .unique()
            .tolist()
        )
        raise ValueError(
            "SubjectsInfo.xlsx contains duplicate subject Codes: "
            f"{duplicate_ids[:10]}"
        )

    metadata_by_id = metadata.set_index("Code")

    records = []

    for image_path in sorted(image_root.glob("*.png")):
        match = FILENAME_PATTERN.match(image_path.name)
        if match is None:
            continue

        subject_id, muscle_code, side_code, image_index = map(
            int, match.groups()
        )

        mask_path = mask_root / image_path.name
        grade_column = f"{muscle_code:03d}_{side_code:02d}"

        grade = None
        metadata_available = subject_id in metadata_by_id.index

        if metadata_available and grade_column in metadata_by_id.columns:
            grade = metadata_by_id.loc[subject_id, grade_column]

        records.append(
            {
                "filename": image_path.name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_exists": mask_path.is_file(),
                "subject_id": subject_id,
                "muscle_code": f"{muscle_code:03d}",
                "side_code": side_code,
                "side": {0: "left", 1: "right"}.get(
                    side_code, "unknown"
                ),
                "image_index": image_index,
                "heckmatt_grade": grade,
                "metadata_available": metadata_available,
            }
        )

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)

    if manifest.empty:
        raise ValueError(
            "No PNG filenames matched the expected pattern "
            "'subject_muscle_side_index.png'."
        )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_file, index=False)

    return manifest
