"""Audit saved Dataset B radiomics feature CSVs against canonical files.

Checks the train/validation/test feature tables against the audited labelled
image manifest and the approved subject-level split. This module only reads
canonical inputs and never creates or modifies the subject split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd


DEFAULT_FEATURE_FILES = {
    "train": "Dataset_B_Radiomics_Training_Features.csv",
    "validation": "Dataset_B_Radiomics_Validation_Features.csv",
    "test": "Dataset_B_Radiomics_Test_Features.csv",
}
DEFAULT_EXPECTED_COUNTS = {
    "train": 3342,
    "validation": 741,
    "test": 644,
}
VALID_PARTITIONS = ("train", "validation", "test")

MANIFEST_COLUMNS = {
    "filename", "subject_id", "muscle_code", "side_code", "heckmatt_grade"
}
SPLIT_COLUMNS = {"subject_id", "partition"}
FEATURE_COLUMNS = {
    "filename", "subject_id", "muscle_code", "side_code",
    "group_id", "heckmatt_grade", "partition"
}
METADATA_COLUMNS = (
    "subject_id", "muscle_code", "side_code", "group_id",
    "heckmatt_grade", "partition"
)


def _normalise_values(values: pd.Series) -> pd.Series:
    """Compare numeric IDs/grades consistently across CSV dtype inference."""
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype("string").fillna("<NA>")
    mask = numeric.notna()
    normalized.loc[mask] = numeric.loc[mask].map(
        lambda value: str(int(value)) if float(value).is_integer()
        else format(float(value), ".12g")
    )
    return normalized


def _load_canonical_records(
    manifest_path: str | Path,
    split_path: str | Path,
) -> pd.DataFrame:
    """Load labelled manifest rows and attach their approved partition."""
    manifest_file = Path(manifest_path).expanduser()
    split_file = Path(split_path).expanduser()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Canonical manifest not found: {manifest_file}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Canonical split not found: {split_file}")

    manifest = pd.read_csv(manifest_file)
    split = pd.read_csv(split_file)

    missing_manifest = MANIFEST_COLUMNS.difference(manifest.columns)
    missing_split = SPLIT_COLUMNS.difference(split.columns)
    if missing_manifest:
        raise ValueError(
            "Manifest is missing columns: " + ", ".join(sorted(missing_manifest))
        )
    if missing_split:
        raise ValueError(
            "Split is missing columns: " + ", ".join(sorted(missing_split))
        )
    if manifest["filename"].isna().any():
        raise ValueError("Manifest contains missing filenames.")
    if manifest["filename"].astype(str).duplicated().any():
        raise ValueError("Manifest contains duplicate filenames.")
    if split["subject_id"].isna().any() or split["partition"].isna().any():
        raise ValueError("Canonical split contains missing values.")
    if split["subject_id"].duplicated().any():
        raise ValueError("Canonical split must contain one row per subject.")

    split["partition"] = split["partition"].astype(str).str.strip().str.lower()
    unexpected = set(split["partition"].unique()) - set(VALID_PARTITIONS)
    if unexpected:
        raise ValueError(
            f"Unexpected canonical partition values: {sorted(unexpected)}"
        )

    labelled = manifest.loc[manifest["heckmatt_grade"].notna()].copy()
    if labelled.empty:
        raise ValueError("Manifest contains no labelled records.")
    grades = pd.to_numeric(labelled["heckmatt_grade"], errors="coerce")
    if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
        raise ValueError("Labelled Heckmatt grades must be in {1, 2, 3, 4}.")
    labelled["heckmatt_grade"] = grades.astype(int)

    if "group_id" not in labelled.columns:
        labelled["group_id"] = (
            labelled["subject_id"].astype(str) + "_"
            + labelled["muscle_code"].astype(str) + "_"
            + labelled["side_code"].astype(str)
        )

    labelled = labelled.merge(
        split[["subject_id", "partition"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = labelled["_merge"] != "both"
    if unmatched.any():
        subjects = (
            labelled.loc[unmatched, "subject_id"]
            .drop_duplicates().astype(str).tolist()
        )
        raise ValueError(
            "Manifest subjects missing from canonical split; "
            f"examples: {subjects[:10]}"
        )
    return labelled.drop(columns="_merge")


def validate_radiomics_feature_partitions(
    manifest_path: str | Path,
    split_path: str | Path,
    feature_dir: str | Path,
    *,
    feature_files: Optional[Mapping[str, str]] = None,
    expected_counts: Optional[Mapping[str, int]] = DEFAULT_EXPECTED_COUNTS,
    report_path: Optional[str | Path] = None,
    raise_on_failure: bool = False,
) -> pd.DataFrame:
    """Compare the three feature CSVs with the canonical manifest and split.

    The report provides expected and actual row counts, missing/unexpected
    filenames, duplicate filenames, missing values, metadata mismatches,
    feature-column counts, subject/group counts, and PASS/FAIL status.

    ``expected_counts=None`` disables fixed record-count checks for subsets.
    ``raise_on_failure=True`` raises after producing the report if any
    partition fails.
    """
    canonical = _load_canonical_records(manifest_path, split_path)
    feature_root = Path(feature_dir).expanduser()
    if not feature_root.is_dir():
        raise NotADirectoryError(
            f"Feature directory not found: {feature_root}"
        )

    filenames = dict(DEFAULT_FEATURE_FILES if feature_files is None else feature_files)
    if set(filenames) != set(VALID_PARTITIONS):
        raise ValueError(
            "feature_files must define exactly train, validation, and test."
        )

    report_rows = []
    canonical_subject_sets = {}

    for partition in VALID_PARTITIONS:
        expected = canonical.loc[canonical["partition"] == partition].copy()
        canonical_subject_sets[partition] = set(
            expected["subject_id"].astype(str)
        )

        csv_path = feature_root / filenames[partition]
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"{partition} features not found: {csv_path}"
            )

        features = pd.read_csv(csv_path)
        missing_columns = FEATURE_COLUMNS.difference(features.columns)
        if missing_columns:
            raise ValueError(
                f"{partition} features are missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        actual_names = features["filename"].astype(str)
        expected_names = expected["filename"].astype(str)
        duplicate_names = int(actual_names.duplicated().sum())
        missing_names = set(expected_names) - set(actual_names)
        extra_names = set(actual_names) - set(expected_names)

        expected_lookup = expected.assign(
            _key=expected["filename"].astype(str)
        ).set_index("_key")
        actual_lookup = (
            features.assign(_key=actual_names)
            .drop_duplicates("_key", keep="first")
            .set_index("_key")
        )
        common = sorted(set(expected_lookup.index) & set(actual_lookup.index))

        mismatch_counts = {}
        for column in METADATA_COLUMNS:
            left = _normalise_values(
                expected_lookup.loc[common, column].reset_index(drop=True)
            )
            right = _normalise_values(
                actual_lookup.loc[common, column].reset_index(drop=True)
            )
            mismatch_counts[column] = int((left != right).sum())

        feature_names = [
            str(column) for column in features.columns
            if str(column).startswith("original_")
        ]
        expected_count_ok = (
            expected_counts is None
            or partition not in expected_counts
            or len(features) == int(expected_counts[partition])
        )
        subject_match = (
            set(features["subject_id"].astype(str))
            == canonical_subject_sets[partition]
        )
        metadata_mismatches = int(sum(mismatch_counts.values()))
        missing_values = int(features.isna().sum().sum())

        passed = (
            len(features) == len(expected)
            and expected_count_ok
            and not missing_names
            and not extra_names
            and duplicate_names == 0
            and missing_values == 0
            and metadata_mismatches == 0
            and subject_match
        )

        row = {
            "partition": partition,
            "expected_records": int(len(expected)),
            "actual_records": int(len(features)),
            "missing_filenames": int(len(missing_names)),
            "unexpected_filenames": int(len(extra_names)),
            "duplicate_filenames": duplicate_names,
            "missing_values": missing_values,
            "metadata_mismatches": metadata_mismatches,
            "feature_columns": len(feature_names),
            "subjects": int(features["subject_id"].nunique()),
            "groups": int(features["group_id"].nunique()),
            "canonical_subjects": len(canonical_subject_sets[partition]),
            "subject_set_matches": bool(subject_match),
            "status": "PASS" if passed else "FAIL",
        }
        for column, count in mismatch_counts.items():
            row[f"{column}_mismatches"] = count
        report_rows.append(row)

    report = pd.DataFrame(report_rows)

    for index, first in enumerate(VALID_PARTITIONS):
        for second in VALID_PARTITIONS[index + 1:]:
            overlap = len(
                canonical_subject_sets[first] & canonical_subject_sets[second]
            )
            report[f"{first}_vs_{second}_subject_overlap"] = overlap
            if overlap:
                report["status"] = "FAIL"

    if report_path is not None:
        destination = Path(report_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(destination, index=False)

    if raise_on_failure and not report["status"].eq("PASS").all():
        failed = report.loc[report["status"] != "PASS", "partition"].tolist()
        raise ValueError(
            "Radiomics feature audit failed for: " + ", ".join(failed)
        )

    return report
