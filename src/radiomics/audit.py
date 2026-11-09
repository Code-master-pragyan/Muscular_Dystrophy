
"""Radiomics provenance, image metadata, and feature-schema audits.

This module converts the final Dataset B radiomics investigation cells into
reusable diagnostics. It provides:
1. A keyword/context search through extraction scripts and configuration files.
2. An image-mask metadata inspection for selected ultrasound examples.
3. A feature schema audit across train, validation, and test CSV tables.
4. Group consistency checks for subject, muscle, side, and Heckmatt grade.

The module does not modify images, masks, feature CSVs, or model inputs.
It is intended for provenance and quality auditing.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")

DEFAULT_EXTRACTION_KEYWORDS = (
    "RadiomicsFeatureExtractor",
    "binWidth",
    "resampledPixelSpacing",
    "normalize",
    "normalizeScale",
    "force2D",
    "imageType",
    "featureClass",
    "correctMask",
    "interpolator",
    "label=",
    "extractor.execute",
)

DEFAULT_SAMPLE_FILENAMES = (
    "00006_001_00_1.png",
    "00006_004_00_1.png",
    "00006_008_00_1.png",
)

DEFAULT_METADATA_COLUMNS = (
    "filename",
    "subject_id",
    "muscle_code",
    "side_code",
    "side",
    "group_id",
    "heckmatt_grade",
    "partition",
)

EXPECTED_FEATURE_CLASS_COUNTS = {
    "firstorder": 18,
    "glcm": 24,
    "glrlm": 16,
    "glszm": 16,
    "gldm": 14,
    "ngtdm": 5,
    "shape2D": 9,
}

EXPECTED_RECORD_COUNTS = {
    "train": 3342,
    "validation": 741,
    "test": 644,
}

EXPECTED_GROUP_COUNTS = {
    "train": 1029,
    "validation": 231,
    "test": 208,
}


def search_extraction_configuration(
    search_dirs: Sequence[str | Path],
    *,
    keywords: Sequence[str] = DEFAULT_EXTRACTION_KEYWORDS,
    extensions: Sequence[str] = (".py", ".yaml", ".yml", ".json"),
    context_before: int = 1,
    context_after: int = 2,
) -> pd.DataFrame:
    """Search script/config files for radiomics settings and nearby context.

    This is a text search only. It reports matching file paths, line numbers,
    keywords, and surrounding lines; it does not parse or execute the files.
    """
    if not keywords:
        raise ValueError("At least one search keyword is required.")

    if context_before < 0 or context_after < 0:
        raise ValueError("Context line counts cannot be negative.")

    allowed_extensions = {
        extension.lower() if extension.startswith(".")
        else "." + extension.lower()
        for extension in extensions
    }

    lowered_keywords = [
        str(keyword).lower()
        for keyword in keywords
    ]

    rows = []

    for directory in search_dirs:
        root = Path(directory).expanduser()

        if not root.exists():
            rows.append({
                "file_path": str(root),
                "line_number": None,
                "keyword": None,
                "context": "",
                "status": "DIRECTORY_NOT_FOUND",
            })
            continue

        if not root.is_dir():
            raise NotADirectoryError(
                f"Search path is not a directory: {root}"
            )

        for file_path in sorted(root.rglob("*")):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in allowed_extensions:
                continue

            try:
                lines = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines()

            except OSError as exc:
                rows.append({
                    "file_path": str(file_path),
                    "line_number": None,
                    "keyword": None,
                    "context": str(exc),
                    "status": "READ_ERROR",
                })
                continue

            for index, line in enumerate(lines):

                matching_keywords = [
                    keyword
                    for keyword in lowered_keywords
                    if keyword in line.lower()
                ]

                if not matching_keywords:
                    continue

                start = max(0, index - context_before)
                end = min(
                    len(lines),
                    index + context_after + 1,
                )

                context = "\n".join(lines[start:end])

                for keyword in matching_keywords:

                    rows.append({
                        "file_path": str(file_path),
                        "line_number": index + 1,
                        "keyword": keyword,
                        "context": context,
                        "status": "MATCH",
                    })

    return pd.DataFrame(
        rows,
        columns=[
            "file_path",
            "line_number",
            "keyword",
            "context",
            "status",
        ],
    )


def inspect_image_mask_metadata(
    image_dir: str | Path,
    mask_dir: str | Path,
    filenames: Sequence[str] = DEFAULT_SAMPLE_FILENAMES,
    *,
    foreground_label: int = 1,
) -> pd.DataFrame:
    """Inspect SimpleITK metadata and masked image intensity for samples.

    The ROI intensity statistics use:
        image_array[mask_array == foreground_label]

    This matches the notebook's processed-mask assumption, where foreground
    pixels are encoded as 1. Original masks encoded as 255 must be normalized
    before using the default foreground_label.
    """
    if not filenames:
        raise ValueError(
            "At least one sample filename is required."
        )

    image_root = Path(image_dir).expanduser()
    mask_root = Path(mask_dir).expanduser()

    if not image_root.is_dir():
        raise NotADirectoryError(
            f"Image directory not found: {image_root}"
        )

    if not mask_root.is_dir():
        raise NotADirectoryError(
            f"Mask directory not found: {mask_root}"
        )

    try:
        import SimpleITK as sitk

    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for image-mask metadata inspection. "
            "Install it with: pip install SimpleITK"
        ) from exc

    rows = []

    for filename in filenames:

        image_path = image_root / str(filename)
        mask_path = mask_root / str(filename)

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

        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        if image_array.shape != mask_array.shape:
            raise ValueError(
                f"Image/mask array shape mismatch for {filename}: "
                f"{image_array.shape} versus {mask_array.shape}."
            )

        roi_values = image_array[
            mask_array == foreground_label
        ]

        roi_pixel_count = int(roi_values.size)

        geometry_matches = (
            image.GetDimension() == mask.GetDimension()
            and image.GetSize() == mask.GetSize()
            and np.allclose(
                image.GetSpacing(),
                mask.GetSpacing(),
            )
            and np.allclose(
                image.GetOrigin(),
                mask.GetOrigin(),
            )
            and np.allclose(
                image.GetDirection(),
                mask.GetDirection(),
            )
        )

        rows.append({
            "filename": str(filename),

            "image_dimension": int(image.GetDimension()),
            "image_size": tuple(image.GetSize()),
            "image_spacing": tuple(image.GetSpacing()),
            "image_origin": tuple(image.GetOrigin()),
            "image_direction": tuple(image.GetDirection()),
            "image_pixel_type": image.GetPixelIDTypeAsString(),

            "mask_dimension": int(mask.GetDimension()),
            "mask_size": tuple(mask.GetSize()),
            "mask_spacing": tuple(mask.GetSpacing()),
            "mask_origin": tuple(mask.GetOrigin()),
            "mask_direction": tuple(mask.GetDirection()),

            "image_mask_geometry_matches": bool(
                geometry_matches
            ),

            "foreground_label": int(foreground_label),
            "roi_pixel_count": roi_pixel_count,

            "roi_minimum": (
                float(np.min(roi_values))
                if roi_pixel_count
                else np.nan
            ),

            "roi_maximum": (
                float(np.max(roi_values))
                if roi_pixel_count
                else np.nan
            ),

            "roi_mean": (
                float(np.mean(roi_values))
                if roi_pixel_count
                else np.nan
            ),

            "roi_standard_deviation": (
                float(np.std(roi_values))
                if roi_pixel_count
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def audit_extraction_script_feature_classes(
    script_paths: Sequence[str | Path],
) -> pd.DataFrame:
    """Report feature-class and extractor configuration snippets in scripts.

    This function searches the expressions inspected in the notebook.
    The output is an evidence table, not a definitive parser of active
    configuration settings.
    """

    patterns = {
        "feature_classes_assignment": re.compile(
            r"feature_classes\s*=\s*\[.*?\]",
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "enable_feature_class": re.compile(
            r"enableFeatureClassByName\s*\([^\n]*",
            flags=re.IGNORECASE,
        ),

        "enable_all_features": re.compile(
            r"enableAllFeatures\s*\([^\n]*",
            flags=re.IGNORECASE,
        ),

        "disable_all_features": re.compile(
            r"disableAllFeatures\s*\([^\n]*",
            flags=re.IGNORECASE,
        ),

        "enable_image_type": re.compile(
            r"enableImageTypeByName\s*\([^\n]*",
            flags=re.IGNORECASE,
        ),

        "extractor_initialization": re.compile(
            r"RadiomicsFeatureExtractor\s*\([^\n]*",
            flags=re.IGNORECASE,
        ),
    }

    rows = []

    for script_path in script_paths:

        path = Path(script_path).expanduser()

        if not path.is_file():

            rows.append({
                "script_path": str(path),
                "pattern": None,
                "matched_text": "File not found",
                "status": "NOT_FOUND",
            })

            continue

        content = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for pattern_name, pattern in patterns.items():

            matches = pattern.findall(content)

            if not matches:

                rows.append({
                    "script_path": str(path),
                    "pattern": pattern_name,
                    "matched_text": "",
                    "status": "NO_MATCH",
                })

                continue

            for match in matches:

                rows.append({
                    "script_path": str(path),
                    "pattern": pattern_name,
                    "matched_text": str(match).strip()[:3000],
                    "status": "MATCH",
                })

    return pd.DataFrame(
        rows,
        columns=[
            "script_path",
            "pattern",
            "matched_text",
            "status",
        ],
    )


def _feature_class_from_column(
    column_name: str,
) -> Optional[str]:
    """Extract the PyRadiomics class name from an original_* column."""

    prefix = "original_"

    if not str(column_name).startswith(prefix):
        return None

    remainder = str(column_name)[len(prefix):]

    class_name = remainder.split(
        "_",
        maxsplit=1,
    )[0]

    if class_name.lower() == "shape2d":
        return "shape2D"

    return class_name.lower()


def audit_radiomics_feature_partitions(
    feature_tables: Mapping[str, pd.DataFrame],
    *,
    expected_record_counts: Optional[Mapping[str, int]] = EXPECTED_RECORD_COUNTS,
    expected_group_counts: Optional[Mapping[str, int]] = EXPECTED_GROUP_COUNTS,
    expected_feature_class_counts: Optional[Mapping[str, int]] = (
        EXPECTED_FEATURE_CLASS_COUNTS
    ),
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    raise_on_failure: bool = False,
) -> pd.DataFrame:
    """Audit feature schema, ordering, partition labels, and group consistency.

    Checks:
    - Feature counts by PyRadiomics class.
    - Duplicate column names and filename records.
    - Missing and infinite feature values.
    - Feature-column order consistency across partitions.
    - Group-level consistency for subject, muscle, side, and grade.
    - Expected image-record and group counts.
    """

    missing_partitions = set(PARTITIONS).difference(
        feature_tables
    )

    if missing_partitions:
        raise ValueError(
            "feature_tables must contain train, validation, and test; "
            f"missing: {sorted(missing_partitions)}."
        )

    reports = []
    feature_orders = {}
    failures = {}

    for partition in PARTITIONS:

        dataframe = feature_tables[partition]

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"{partition} must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                f"{partition} feature table is empty."
            )

        required = {
            "filename",
            "subject_id",
            "muscle_code",
            "side_code",
            "group_id",
            "heckmatt_grade",
            "partition",
        }

        missing_columns = required.difference(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{partition} is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        all_columns = list(dataframe.columns)

        duplicate_column_names = [
            name
            for name, count in Counter(all_columns).items()
            if count > 1
        ]

        feature_columns = [
            str(column)
            for column in dataframe.columns
            if str(column).startswith("original_")
        ]

        feature_orders[partition] = feature_columns

        class_counts = Counter(
            _feature_class_from_column(column)
            for column in feature_columns
        )

        missing_values = int(
            dataframe[feature_columns]
            .isna()
            .sum()
            .sum()
        ) if feature_columns else 0

        numeric_features = (
            dataframe[feature_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            if feature_columns
            else pd.DataFrame(index=dataframe.index)
        )

        non_numeric_or_missing = int(
            numeric_features.isna().sum().sum()
        )

        if feature_columns:

            feature_array = numeric_features.to_numpy(
                dtype=np.float64
            )

            infinite_values = int(
                np.isinf(feature_array).sum()
            )

        else:
            infinite_values = 0

        duplicate_filenames = int(
            dataframe["filename"]
            .astype(str)
            .duplicated()
            .sum()
        )

        partition_values = (
            dataframe["partition"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        partition_label_ok = bool(
            partition_values.eq(partition).all()
        )

        group_columns = [
            "subject_id",
            "muscle_code",
            "side_code",
            "heckmatt_grade",
        ]

        group_nunique = (
            dataframe.groupby(
                "group_id",
                dropna=False,
            )[group_columns]
            .nunique(dropna=False)
        )

        conflicting_group_counts = {
            column: int(
                (group_nunique[column] > 1).sum()
            )
            for column in group_columns
        }

        groups = int(
            dataframe["group_id"].nunique()
        )

        expected_records_ok = (
            expected_record_counts is None
            or partition not in expected_record_counts
            or len(dataframe)
            == int(expected_record_counts[partition])
        )

        expected_groups_ok = (
            expected_group_counts is None
            or partition not in expected_group_counts
            or groups == int(expected_group_counts[partition])
        )

        class_count_ok = (
            expected_feature_class_counts is None
            or all(
                class_counts.get(class_name, 0) == expected_count
                for class_name, expected_count
                in expected_feature_class_counts.items()
            )
        )

        group_consistent = not any(
            conflicting_group_counts.values()
        )

        no_invalid_features = (
            non_numeric_or_missing == 0
            and infinite_values == 0
        )

        no_duplicate_records = (
            duplicate_filenames == 0
        )

        passed = (
            expected_records_ok
            and expected_groups_ok
            and class_count_ok
            and group_consistent
            and no_invalid_features
            and no_duplicate_records
            and partition_label_ok
            and not duplicate_column_names
        )

        row = {
            "partition": partition,
            "records": int(len(dataframe)),
            "groups": groups,
            "total_columns": int(len(dataframe.columns)),

            "metadata_columns_present": int(
                sum(
                    column in dataframe.columns
                    for column in metadata_columns
                )
            ),

            "radiomics_feature_count": int(
                len(feature_columns)
            ),

            "duplicate_column_names": duplicate_column_names,
            "duplicate_filenames": duplicate_filenames,
            "missing_feature_values": missing_values,

            "non_numeric_or_missing_feature_values": (
                non_numeric_or_missing
            ),

            "infinite_feature_values": infinite_values,

            "partition_label_ok": partition_label_ok,
            "expected_record_count_ok": bool(expected_records_ok),
            "expected_group_count_ok": bool(expected_groups_ok),
            "feature_class_count_ok": bool(class_count_ok),
            "group_consistent": bool(group_consistent),

            "groups_with_conflicting_subjects": (
                conflicting_group_counts["subject_id"]
            ),

            "groups_with_conflicting_muscles": (
                conflicting_group_counts["muscle_code"]
            ),

            "groups_with_conflicting_sides": (
                conflicting_group_counts["side_code"]
            ),

            "groups_with_conflicting_grades": (
                conflicting_group_counts["heckmatt_grade"]
            ),

            "status": "PASS" if passed else "FAIL",
        }

        for class_name in (
            expected_feature_class_counts
            if expected_feature_class_counts is not None
            else class_counts.keys()
        ):
            row[f"{class_name}_features"] = int(
                class_counts.get(class_name, 0)
            )

        reports.append(row)
        failures[partition] = not passed

    # Check that all three partitions use identical feature columns
    # in identical order.

    reference_order = feature_orders["train"]

    order_matches = {
        partition: (
            feature_orders[partition] == reference_order
        )
        for partition in PARTITIONS
    }

    report = pd.DataFrame(reports)

    report["feature_order_matches_train"] = [
        order_matches[partition]
        for partition in PARTITIONS
    ]

    for index, first in enumerate(PARTITIONS):

        for second in PARTITIONS[index + 1:]:

            report[
                f"{first}_vs_{second}_feature_order_match"
            ] = (
                feature_orders[first]
                == feature_orders[second]
            )

    if not all(order_matches.values()):
        report["status"] = "FAIL"

    if raise_on_failure and not report["status"].eq("PASS").all():

        failed = report.loc[
            report["status"] != "PASS",
            "partition",
        ].tolist()

        raise ValueError(
            "Radiomics feature partition audit failed for: "
            + ", ".join(failed)
        )

    return report


def save_radiomics_provenance_audit(
    configuration_search: pd.DataFrame,
    script_feature_classes: pd.DataFrame,
    image_mask_metadata: pd.DataFrame,
    feature_partition_report: pd.DataFrame,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save the four provenance and metadata audit tables as CSV files."""

    artifacts = {
        "configuration_search": (
            configuration_search,
            "Radiomics_Extraction_Configuration_Search.csv",
        ),

        "script_feature_classes": (
            script_feature_classes,
            "Radiomics_Script_Feature_Class_Audit.csv",
        ),

        "image_mask_metadata": (
            image_mask_metadata,
            "Ultrasound_Image_Mask_Metadata_Audit.csv",
        ),

        "feature_partition_report": (
            feature_partition_report,
            "Radiomics_Feature_Partition_Schema_Audit.csv",
        ),
    }

    destination = Path(output_dir).expanduser()
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        key: destination / filename
        for key, (_, filename) in artifacts.items()
    }

    existing = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing audit files: "
            + ", ".join(str(path) for path in existing)
            + ". Pass overwrite=True to replace them."
        )

    for key, (dataframe, _) in artifacts.items():

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"{key} must be a pandas DataFrame."
            )

        dataframe.to_csv(
            paths[key],
            index=False,
        )

    return paths
