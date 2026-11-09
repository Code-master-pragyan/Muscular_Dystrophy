"""Small subject-level PyRadiomics pilot used to smoke-test extraction.

This module converts the 10-subject pilot in Step_2_Dataset_Audit.ipynb
(cells 88-89) into reusable functions. It selects at most one labelled
training image per subject, then extracts features from the corresponding
processed mask. It does not replace the full-dataset extractor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


PILOT_REQUIRED_COLUMNS = {
    "filename",
    "image_path",
    "subject_id",
    "muscle_code",
    "side_code",
    "side",
    "heckmatt_grade",
    "partition",
    "image_index",
}
PILOT_METADATA_COLUMNS = [
    "filename",
    "subject_id",
    "muscle_code",
    "side",
    "group_id",
    "heckmatt_grade",
    "partition",
]


def select_subject_pilot(
    labelled_df: pd.DataFrame,
    *,
    n_subjects: int = 10,
    partition: str = "train",
) -> pd.DataFrame:
    """Select the first image from up to ``n_subjects`` distinct subjects.

    Selection follows the notebook order exactly: filter the requested
    partition, sort by subject, muscle, side code, and image index, keep the
    first record for each subject, then take the first ``n_subjects`` rows.
    """
    if not isinstance(labelled_df, pd.DataFrame):
        raise TypeError("labelled_df must be a pandas DataFrame.")
    missing = PILOT_REQUIRED_COLUMNS.difference(labelled_df.columns)
    if missing:
        raise ValueError(
            "labelled_df is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1.")
    if not isinstance(partition, str) or not partition.strip():
        raise ValueError("partition must be a non-empty string.")

    selected = (
        labelled_df.loc[labelled_df["partition"].astype(str) == partition]
        .sort_values(
            ["subject_id", "muscle_code", "side_code", "image_index"]
        )
        .drop_duplicates(subset="subject_id", keep="first")
        .head(n_subjects)
        .copy()
    )
    return selected


def extract_subject_pilot_features(
    pilot_df: pd.DataFrame,
    processed_masks_dir: str | Path,
    *,
    extractor: Optional[Any] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features for the selected pilot records.

    Returns ``(features, errors)``. The feature table contains the same
    identifying metadata as the notebook and all non-diagnostic scalar
    PyRadiomics outputs. Failed records are returned in a separate error
    table instead of stopping the remaining pilot records.

    Pass an already configured PyRadiomics extractor to ``extractor`` to
    reuse it. Otherwise the project's configured ``create_extractor`` helper
    is imported lazily from :mod:`src.radiomics.extraction`.
    """
    if not isinstance(pilot_df, pd.DataFrame):
        raise TypeError("pilot_df must be a pandas DataFrame.")
    required = PILOT_REQUIRED_COLUMNS.difference({"image_index"})
    missing = required.difference(pilot_df.columns)
    if missing:
        raise ValueError(
            "pilot_df is missing required columns: "
            + ", ".join(sorted(missing))
        )

    masks_dir = Path(processed_masks_dir)
    if not masks_dir.is_dir():
        raise NotADirectoryError(
            f"Processed masks directory not found: {masks_dir}"
        )

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for radiomics pilot extraction. "
            "Install the project dependencies first."
        ) from exc

    if extractor is None:
        try:
            from .extraction import create_extractor
        except ImportError:
            # Allows direct execution/import when the repository root is not
            # exposed as the ``src`` package, while retaining package usage.
            from extraction import create_extractor  # type: ignore
        extractor = create_extractor()

    feature_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, str]] = []

    for _, row in pilot_df.iterrows():
        filename = str(row["filename"])
        image_path = Path(str(row["image_path"]))
        mask_path = masks_dir / filename

        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Processed mask not found: {mask_path}")

            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))
            image = sitk.Cast(image, sitk.sitkFloat32)

            if image.GetDimension() != 2 or mask.GetDimension() != 2:
                raise ValueError("Expected 2D image and mask.")
            if image.GetSize() != mask.GetSize():
                raise ValueError("Image and mask sizes do not match.")

            result = extractor.execute(image, mask, label=1)
            record: dict[str, Any] = {
                "filename": filename,
                "subject_id": row["subject_id"],
                "muscle_code": row["muscle_code"],
                "side": row["side"],
                "group_id": (
                    f"{row['subject_id']}_{row['muscle_code']}_{row['side_code']}"
                ),
                "heckmatt_grade": int(row["heckmatt_grade"]),
                "partition": row["partition"],
            }

            for key, value in result.items():
                if not key.startswith("diagnostics_"):
                    array = np.asarray(value)
                    if array.size != 1:
                        raise ValueError(
                            f"Feature {key} is not a single numeric value."
                        )
                    record[key] = float(array.item())

            feature_rows.append(record)

        except Exception as exc:
            error_rows.append(
                {
                    "filename": filename,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    features = pd.DataFrame(feature_rows)
    errors = pd.DataFrame(error_rows, columns=["filename", "error"])
    return features, errors


def summarize_subject_pilot(
    features: pd.DataFrame,
    errors: pd.DataFrame,
) -> dict[str, int]:
    """Summarize extraction counts and non-finite feature values."""
    if not isinstance(features, pd.DataFrame) or not isinstance(errors, pd.DataFrame):
        raise TypeError("features and errors must both be pandas DataFrames.")

    feature_columns = [
        column
        for column in features.columns
        if isinstance(column, str) and column.startswith("original_")
    ]
    if feature_columns:
        numeric = features[feature_columns].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        missing_values = int(np.isnan(values).sum())
        infinite_values = int(np.isinf(values).sum())
    else:
        missing_values = 0
        infinite_values = 0

    return {
        "successful_extractions": int(len(features)),
        "extraction_errors": int(len(errors)),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(feature_columns)),
        "distinct_subjects": (
            int(features["subject_id"].nunique())
            if "subject_id" in features.columns
            else 0
        ),
        "missing_feature_values": missing_values,
        "infinite_feature_values": infinite_values,
    }
