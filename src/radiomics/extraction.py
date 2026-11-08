
"""Configurable PyRadiomics extraction for Dataset B FSHD ultrasound images.

This reusable implementation records the documented Dataset B settings.
It does not establish which historical extraction script generated the
archived feature CSV files.

Accepts binary masks encoded as {0, 1} or {0, 255}, normalizing the
foreground to label 1 for PyRadiomics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


METADATA_COLUMNS = [
    "filename",
    "subject_id",
    "muscle_code",
    "side_code",
    "side",
    "group_id",
    "heckmatt_grade",
    "partition",
]

REQUIRED_MANIFEST_COLUMNS = {
    "filename",
    "image_path",
    "subject_id",
    "muscle_code",
    "side_code",
    "heckmatt_grade",
}

VALID_PARTITIONS = {"train", "validation", "test"}

FEATURE_CLASSES = [
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
    "shape2D",
]


def create_extractor():
    """Create the PyRadiomics extractor using documented settings."""
    settings = {
        "binWidth": 25,
        "label": 1,
        "normalize": False,
        "resampledPixelSpacing": None,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    for feature_class in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class)

    extractor.enableImageTypeByName("Original")
    return extractor


def _prepare_manifest(
    manifest: pd.DataFrame,
    split_path: Optional[str] = None,
) -> pd.DataFrame:
    """Validate metadata, derive group_id, and optionally attach the split."""
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing))
        )

    prepared = manifest.copy()

    for column in (
        "filename",
        "image_path",
        "subject_id",
        "muscle_code",
        "side_code",
    ):
        if prepared[column].isna().any():
            raise ValueError(
                f"Manifest contains missing values in {column}."
            )

    if "group_id" not in prepared.columns:
        prepared["group_id"] = (
            prepared["subject_id"].astype(str)
            + "_"
            + prepared["muscle_code"].astype(str)
            + "_"
            + prepared["side_code"].astype(str)
        )

    if split_path is not None:
        if "partition" in prepared.columns:
            raise ValueError(
                "The manifest already contains 'partition'. "
                "Omit split_path or use a manifest without an attached split."
            )

        split_file = Path(split_path)

        if not split_file.is_file():
            raise FileNotFoundError(
                f"Subject split not found: {split_file}"
            )

        split_df = pd.read_csv(split_file)

        required_split_columns = {"subject_id", "partition"}
        missing_split = required_split_columns.difference(split_df.columns)

        if missing_split:
            raise ValueError(
                "Subject split is missing required columns: "
                + ", ".join(sorted(missing_split))
            )

        if (
            split_df["subject_id"].isna().any()
            or split_df["partition"].isna().any()
        ):
            raise ValueError(
                "Subject split contains missing subject_id/partition."
            )

        if split_df["subject_id"].duplicated().any():
            raise ValueError(
                "Subject split must contain one row per subject."
            )

        split_df["partition"] = (
            split_df["partition"].astype(str).str.strip()
        )

        unexpected = set(split_df["partition"].unique()) - VALID_PARTITIONS

        if unexpected:
            raise ValueError(
                f"Unexpected partition values: {sorted(unexpected)}"
            )

        prepared = prepared.merge(
            split_df[["subject_id", "partition"]],
            on="subject_id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        unmatched = prepared["_merge"] != "both"

        if unmatched.any():
            missing_subjects = (
                prepared.loc[unmatched, "subject_id"]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                "Some manifest subjects are absent from the approved split. "
                f"Examples: {missing_subjects[:10]}"
            )

        prepared = prepared.drop(columns=["_merge"])

    if "partition" in prepared.columns:
        if prepared["partition"].isna().any():
            raise ValueError(
                "Manifest contains missing partition values."
            )

        prepared["partition"] = (
            prepared["partition"].astype(str).str.strip()
        )

        unexpected = (
            set(prepared["partition"].unique()) - VALID_PARTITIONS
        )

        if unexpected:
            raise ValueError(
                f"Unexpected partition values: {sorted(unexpected)}"
            )

    return prepared


def _normalize_binary_mask(
    mask_image: sitk.Image,
    filename: str,
) -> sitk.Image:
    """Normalize binary masks from {0,1} or {0,255} to {0,1}."""
    mask_array = sitk.GetArrayFromImage(mask_image)
    unique_values = set(np.unique(mask_array).tolist())

    if unique_values.issubset({0, 1}):
        binary_array = mask_array.astype(np.uint8)

    elif unique_values.issubset({0, 255}):
        binary_array = (mask_array == 255).astype(np.uint8)

    else:
        raise ValueError(
            f"Mask {filename} has unexpected values "
            f"{sorted(unique_values)}. "
            "Expected {0,1} or {0,255}."
        )

    if not np.any(binary_array == 1):
        raise ValueError(
            f"Mask {filename} contains no foreground pixels."
        )

    normalized = sitk.GetImageFromArray(binary_array)
    normalized.CopyInformation(mask_image)

    return normalized


def _check_matching_geometry(
    image: sitk.Image,
    mask: sitk.Image,
) -> None:
    """Ensure image and mask have matching 2D geometry."""
    if image.GetDimension() != 2 or mask.GetDimension() != 2:
        raise ValueError("Expected 2D image and 2D mask.")

    if image.GetSize() != mask.GetSize():
        raise ValueError("Image and mask sizes do not match.")

    if not np.allclose(image.GetSpacing(), mask.GetSpacing()):
        raise ValueError("Image and mask spacing do not match.")

    if not np.allclose(image.GetOrigin(), mask.GetOrigin()):
        raise ValueError("Image and mask origins do not match.")

    if not np.allclose(image.GetDirection(), mask.GetDirection()):
        raise ValueError("Image and mask directions do not match.")


def extract_features(
    manifest_path: str,
    output_path: str,
    mask_dir: str,
    metadata_columns: Optional[Iterable[str]] = None,
    split_path: Optional[str] = None,
) -> tuple[int, int]:
    """Extract radiomics features for the manifest image records."""
    manifest_file = Path(manifest_path)
    output_file = Path(output_path)
    masks_directory = Path(mask_dir)

    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_file}"
        )

    if not masks_directory.is_dir():
        raise NotADirectoryError(
            f"Mask directory not found: {masks_directory}"
        )

    manifest = pd.read_csv(manifest_file)

    manifest = _prepare_manifest(
        manifest,
        split_path=split_path,
    )

    if metadata_columns is None:
        columns_to_keep = [
            column
            for column in METADATA_COLUMNS
            if column in manifest.columns
        ]
    else:
        columns_to_keep = list(metadata_columns)

    invalid_metadata = set(columns_to_keep).difference(manifest.columns)

    if invalid_metadata:
        raise ValueError(
            "Requested metadata columns are missing: "
            + ", ".join(sorted(invalid_metadata))
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    error_file = output_file.with_name(
        output_file.stem + "_Errors.csv"
    )

    extractor = create_extractor()

    results = []
    errors = []

    for row_index, row in manifest.iterrows():
        filename = str(row["filename"])
        image_path = Path(str(row["image_path"]))
        mask_path = masks_directory / filename

        try:
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))

            _check_matching_geometry(image, mask)

            mask = _normalize_binary_mask(mask, filename)
            image = sitk.Cast(image, sitk.sitkFloat32)

            extracted = extractor.execute(image, mask)

            record = {
                column: row[column]
                for column in columns_to_keep
            }

            record.update(
                {
                    key: float(value)
                    for key, value in extracted.items()
                    if key.startswith("original_")
                }
            )

            results.append(record)

        except Exception as exc:
            errors.append(
                {
                    "row_index": int(row_index),
                    "filename": filename,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    # Save both successful records and error details.
    pd.DataFrame(results).to_csv(output_file, index=False)

    pd.DataFrame(
        errors,
        columns=[
            "row_index",
            "filename",
            "error_type",
            "error",
        ],
    ).to_csv(error_file, index=False)

    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Feature CSV: {output_file}")
    print(f"Error CSV: {error_file}")

    if not results:
        raise RuntimeError(
            "Radiomics extraction produced no successful rows. "
            f"Inspect the error report: {error_file}"
        )

    return len(results), len(errors)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract PyRadiomics features from Dataset B ultrasound images."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the audited image manifest CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the feature output CSV",
    )

    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Directory containing matching binary mask PNG files",
    )

    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Optional path to the existing approved subject-level split CSV. "
            "Use only when the manifest has no partition column."
        ),
    )

    args = parser.parse_args()

    extract_features(
        manifest_path=args.manifest,
        output_path=args.output,
        mask_dir=args.mask_dir,
        split_path=args.split,
    )


if __name__ == "__main__":
    main()
