"""Configurable PyRadiomics extraction for Dataset B FSHD ultrasound images.

This is a cleaned, reusable implementation based on the original
Dataset_B_Radiomics_Extraction.py script.

Important provenance note:
The project also contains a separate checkpoint/resume extraction script.
The existence of this implementation does not establish which script
generated the archived feature CSV files.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional

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
    """Create the PyRadiomics extractor with the original script settings."""
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


def extract_features(
    manifest_path: str,
    output_path: str,
    mask_dir: str,
    metadata_columns: Optional[Iterable[str]] = None,
) -> tuple[int, int]:
    """Extract original-image radiomics features for every manifest row.

    The manifest must contain image_path and filename, plus the metadata
    columns listed in METADATA_COLUMNS. Masks are resolved as
    ``mask_dir / filename``.

    Successful rows and extraction errors are written to separate CSV files.
    Only PyRadiomics feature keys beginning with ``original_`` are retained,
    matching the original function-based extraction script.
    """
    manifest_file = Path(manifest_path)
    output_file = Path(output_path)
    masks_directory = Path(mask_dir)

    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    if not masks_directory.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {masks_directory}")

    manifest = pd.read_csv(manifest_file)
    required_columns = set(METADATA_COLUMNS) | {"image_path"}
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing))
        )

    columns_to_keep = list(metadata_columns or METADATA_COLUMNS)
    invalid_metadata = set(columns_to_keep).difference(manifest.columns)
    if invalid_metadata:
        raise ValueError(
            "Requested metadata columns are missing: "
            + ", ".join(sorted(invalid_metadata))
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    extractor = create_extractor()

    results = []
    errors = []

    for row_index, row in manifest.iterrows():
        filename = str(row["filename"])
        image_path = Path(str(row["image_path"]))
        mask_path = masks_directory / filename

        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))

            if image.GetDimension() != 2 or mask.GetDimension() != 2:
                raise ValueError("Expected 2D image and mask")
            if image.GetSize() != mask.GetSize():
                raise ValueError("Image and mask sizes do not match")

            image = sitk.Cast(image, sitk.sitkFloat32)
            extracted = extractor.execute(image, mask)

            record = {column: row[column] for column in columns_to_keep}
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
                    "error": str(exc),
                }
            )

    pd.DataFrame(results).to_csv(output_file, index=False)

    error_file = output_file.with_name(
        output_file.stem + "_Errors.csv"
    )
    pd.DataFrame(errors).to_csv(error_file, index=False)

    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Feature CSV: {output_file}")
    print(f"Error CSV: {error_file}")

    return len(results), len(errors)


def main():
    parser = argparse.ArgumentParser(
        description="Extract PyRadiomics features from Dataset B ultrasound images."
    )
    parser.add_argument("--manifest", required=True, help="Path to the input manifest CSV")
    parser.add_argument("--output", required=True, help="Path for the feature output CSV")
    parser.add_argument("--mask-dir", required=True, help="Directory containing matching masks")
    args = parser.parse_args()

    extract_features(
        manifest_path=args.manifest,
        output_path=args.output,
        mask_dir=args.mask_dir,
    )


if __name__ == "__main__":
    main()
