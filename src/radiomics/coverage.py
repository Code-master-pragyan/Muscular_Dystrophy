"""Quantify processed-mask ROI coverage for the Dataset B radiomics pilot.

This module converts the coverage diagnostic from Colab cell 90 into a
reusable function. It expects the processed masks used by the notebook,
where foreground ROI pixels are encoded as 1. It deliberately does not
rescale or normalize mask values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "filename",
    "image_path",
    "subject_id",
    "heckmatt_grade",
}


def build_mask_coverage_report(
    pilot_df: pd.DataFrame,
    mask_dir: str | Path,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Create a per-image mask coverage report for a pilot DataFrame.

    Parameters
    ----------
    pilot_df:
        The selected pilot rows, such as ``pilot_10`` from Colab cell 88.
        Must contain filename, image_path, subject_id, and heckmatt_grade.
    mask_dir:
        Directory containing the corresponding processed mask PNG files.
        Each mask is located by matching its filename to the image record.
    output_path:
        Optional CSV path for saving the resulting report.

    Returns
    -------
    pandas.DataFrame
        One row per pilot image with image/mask sizes, ROI coverage
        percentage, and the unique mask labels observed.

    Notes
    -----
    The source notebook counts ROI pixels with ``mask_array == 1``.
    Therefore this function is intended for the processed masks encoded
    as {0, 1}; it does not convert original {0, 255} masks.
    """
    missing = REQUIRED_COLUMNS.difference(pilot_df.columns)
    if missing:
        raise ValueError(
            "Pilot DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if pilot_df.empty:
        raise ValueError("Pilot DataFrame is empty; no masks to inspect.")

    masks_root = Path(mask_dir)
    if not masks_root.is_dir():
        raise NotADirectoryError(
            f"Processed mask directory not found: {masks_root}"
        )

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for mask coverage analysis. "
            "Install it with: pip install SimpleITK"
        ) from exc

    coverage_checks = []

    for row_index, row in pilot_df.iterrows():
        filename = str(row["filename"])
        image_path = Path(str(row["image_path"]))
        mask_path = masks_root / filename

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found at row {row_index}: {image_path}"
            )
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Processed mask not found at row {row_index}: {mask_path}"
            )

        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))

        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        if mask_array.size == 0:
            raise ValueError(f"Processed mask is empty: {mask_path}")

        # Match the original Colab calculation: foreground is label 1.
        roi_pixels = int(np.count_nonzero(mask_array == 1))

        coverage_checks.append(
            {
                "filename": filename,
                "subject_id": row["subject_id"],
                "heckmatt_grade": int(row["heckmatt_grade"]),
                "image_size": tuple(image.GetSize()),
                "mask_size": tuple(mask.GetSize()),
                "mask_coverage_percent": round(
                    100.0 * roi_pixels / mask_array.size, 2
                ),
                "mask_labels": np.unique(mask_array).tolist(),
            }
        )

    coverage_df = pd.DataFrame(coverage_checks)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        coverage_df.to_csv(output_file, index=False)

    return coverage_df


def summarize_mask_coverage(coverage_df: pd.DataFrame) -> dict[str, int]:
    """Return the two summary checks printed by Colab cell 90."""
    required = {"image_size", "mask_size", "mask_coverage_percent"}
    missing = required.difference(coverage_df.columns)
    if missing:
        raise ValueError(
            "Coverage report is missing columns: "
            + ", ".join(sorted(missing))
        )

    dimension_mismatches = int(
        sum(
            image_size != mask_size
            for image_size, mask_size in zip(
                coverage_df["image_size"], coverage_df["mask_size"]
            )
        )
    )
    empty_masks = int(
        (coverage_df["mask_coverage_percent"] == 0).sum()
    )

    return {
        "records_checked": int(len(coverage_df)),
        "dimension_mismatches": dimension_mismatches,
        "empty_masks": empty_masks,
    }
