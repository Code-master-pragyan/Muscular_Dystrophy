"""Join CNN embeddings and radiomics features for Dataset B hybrid fusion.

This module converts the CNN + radiomics feature-matching block from the
original Colab workflow into reusable code. Image records are matched by
filename, every record must have a match in both sources, and shared metadata
must agree before the fused table is returned.

The resulting image-level matrix contains 512 CNN features and 102
radiomics features by default (614 combined feature columns). It does not
aggregate records to subject–muscle–side groups or fit a classifier.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")

COMMON_METADATA_COLUMNS = (
    "subject_id",
    "muscle_code",
    "side_code",
    "group_id",
    "heckmatt_grade",
    "partition",
)

CNN_FEATURE_PREFIX = "cnn_feature_"
RADIOMICS_FEATURE_PREFIX = "original_"


def _normalise_for_comparison(series: pd.Series) -> pd.Series:
    """Normalize numeric metadata to compare values across CSV dtype inference."""
    numeric = pd.to_numeric(series, errors="coerce")
    normalized = series.astype("string").fillna("<NA>")
    numeric_mask = numeric.notna()

    normalized.loc[numeric_mask] = numeric.loc[numeric_mask].map(
        lambda value: (
            str(int(value))
            if float(value).is_integer()
            else format(float(value), ".12g")
        )
    )
    return normalized


def _validate_unique_filenames(
    dataframe: pd.DataFrame,
    source_name: str,
) -> None:
    """Require one row per image filename in each feature table."""
    if "filename" not in dataframe.columns:
        raise ValueError(f"{source_name} table must contain 'filename'.")

    if dataframe["filename"].isna().any():
        raise ValueError(f"{source_name} table contains missing filenames.")

    duplicate_mask = dataframe["filename"].astype(str).duplicated(keep=False)
    if duplicate_mask.any():
        examples = (
            dataframe.loc[duplicate_mask, "filename"]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{source_name} table contains duplicate filenames: {examples}"
        )


def merge_cnn_radiomics_features(
    cnn_features: pd.DataFrame,
    radiomics_features: pd.DataFrame,
    partition: str,
    *,
    expected_cnn_features: Optional[int] = 512,
    expected_radiomics_features: Optional[int] = 102,
    expected_records: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge one approved partition's CNN and radiomics image-level features.

    Parameters
    ----------
    cnn_features:
        CNN embedding DataFrame. Requires filename, common metadata columns,
        and columns prefixed with ``cnn_feature_``.
    radiomics_features:
        Radiomics feature DataFrame. Requires filename, common metadata
        columns, ``side``, and columns prefixed with ``original_``.
    partition:
        One of train, validation, or test. Both input tables must contain
        only this partition.
    expected_cnn_features:
        Expected number of CNN columns. Defaults to 512. Pass None to skip.
    expected_radiomics_features:
        Expected number of radiomics columns. Defaults to 102. Pass None
        to skip.
    expected_records:
        Optional expected number of matched image records for this partition.

    Returns
    -------
    (fused_dataframe, audit)
        The fused image-level DataFrame and a report containing matched
        counts, feature counts, metadata mismatch counts, and side values.

    Notes
    -----
    The original notebook uses an outer merge with ``filename`` as the key,
    asserts every row is matched, compares common metadata from both tables,
    retains one verified copy of each shared field, and keeps ``side`` from
    the radiomics table. This function implements the same checks with
    explicit validation errors instead of bare assertions.
    """
    partition = str(partition).strip().lower()
    if partition not in PARTITIONS:
        raise ValueError(
            f"partition must be one of {PARTITIONS}; received {partition!r}."
        )

    if not isinstance(cnn_features, pd.DataFrame):
        raise TypeError("cnn_features must be a pandas DataFrame.")
    if not isinstance(radiomics_features, pd.DataFrame):
        raise TypeError("radiomics_features must be a pandas DataFrame.")
    if cnn_features.empty or radiomics_features.empty:
        raise ValueError("CNN and radiomics feature tables must not be empty.")

    cnn = cnn_features.copy()
    radiomics = radiomics_features.copy()

    _validate_unique_filenames(cnn, "CNN")
    _validate_unique_filenames(radiomics, "Radiomics")

    required_cnn = {"filename", *COMMON_METADATA_COLUMNS}
    required_radiomics = {
        "filename", *COMMON_METADATA_COLUMNS, "side"
    }

    missing_cnn = required_cnn.difference(cnn.columns)
    missing_radiomics = required_radiomics.difference(radiomics.columns)

    if missing_cnn:
        raise ValueError(
            "CNN table is missing required columns: "
            + ", ".join(sorted(missing_cnn))
        )
    if missing_radiomics:
        raise ValueError(
            "Radiomics table is missing required columns: "
            + ", ".join(sorted(missing_radiomics))
        )

    cnn_feature_columns = sorted(
        column for column in cnn.columns
        if str(column).startswith(CNN_FEATURE_PREFIX)
    )
    radiomics_feature_columns = sorted(
        column for column in radiomics.columns
        if str(column).startswith(RADIOMICS_FEATURE_PREFIX)
    )

    if not cnn_feature_columns:
        raise ValueError("No CNN columns prefixed with 'cnn_feature_' found.")
    if not radiomics_feature_columns:
        raise ValueError("No radiomics columns prefixed with 'original_' found.")

    if (
        expected_cnn_features is not None
        and len(cnn_feature_columns) != expected_cnn_features
    ):
        raise ValueError(
            f"Expected {expected_cnn_features} CNN features, found "
            f"{len(cnn_feature_columns)}."
        )
    if (
        expected_radiomics_features is not None
        and len(radiomics_feature_columns) != expected_radiomics_features
    ):
        raise ValueError(
            f"Expected {expected_radiomics_features} radiomics features, found "
            f"{len(radiomics_feature_columns)}."
        )

    for source_name, dataframe in (
        ("CNN", cnn),
        ("Radiomics", radiomics),
    ):
        if dataframe[list(COMMON_METADATA_COLUMNS)].isna().any().any():
            raise ValueError(
                f"{source_name} table has missing shared metadata values."
            )
        actual_partitions = set(
            dataframe["partition"].astype(str).str.strip().str.lower()
        )
        if actual_partitions != {partition}:
            raise ValueError(
                f"{source_name} table must contain only {partition!r}; "
                f"found {sorted(actual_partitions)}."
            )

    cnn_numeric = cnn[cnn_feature_columns].apply(pd.to_numeric, errors="coerce")
    radiomics_numeric = radiomics[radiomics_feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if cnn_numeric.isna().any().any():
        raise ValueError("CNN feature columns contain missing or non-numeric values.")
    if radiomics_numeric.isna().any().any():
        raise ValueError(
            "Radiomics feature columns contain missing or non-numeric values."
        )
    if not np.isfinite(cnn_numeric.to_numpy(dtype=float)).all():
        raise ValueError("CNN feature columns contain non-finite values.")
    if not np.isfinite(radiomics_numeric.to_numpy(dtype=float)).all():
        raise ValueError("Radiomics feature columns contain non-finite values.")

    cnn.loc[:, cnn_feature_columns] = cnn_numeric
    radiomics.loc[:, radiomics_feature_columns] = radiomics_numeric

    # Outer merge mirrors the source notebook's complete-match audit.
    merged = cnn.merge(
        radiomics,
        on="filename",
        how="outer",
        suffixes=("_cnn", "_rad"),
        indicator=True,
        validate="one_to_one",
    )

    merge_counts = {
        str(key): int(value)
        for key, value in merged["_merge"].value_counts().items()
    }
    unmatched = merged.loc[merged["_merge"] != "both", "filename"].astype(str)
    if not unmatched.empty:
        raise ValueError(
            f"{partition}: CNN/radiomics filename mismatch. "
            f"Unmatched records: {unmatched.head(10).tolist()}."
        )

    metadata_mismatches = {}
    for column in COMMON_METADATA_COLUMNS:
        cnn_column = f"{column}_cnn"
        radiomics_column = f"{column}_rad"

        left = _normalise_for_comparison(merged[cnn_column])
        right = _normalise_for_comparison(merged[radiomics_column])
        mismatch_count = int((left != right).sum())
        metadata_mismatches[column] = mismatch_count

        if mismatch_count:
            examples = merged.loc[
                left != right,
                ["filename", cnn_column, radiomics_column],
            ].head(5)
            raise ValueError(
                f"{partition}: metadata mismatch in {column!r} "
                f"for {mismatch_count} record(s). Examples: "
                f"{examples.to_dict(orient='records')}."
            )

    # Keep one verified shared metadata copy, plus the radiomics side label.
    for column in COMMON_METADATA_COLUMNS:
        merged[column] = merged[f"{column}_cnn"]
        merged.drop(
            columns=[f"{column}_cnn", f"{column}_rad"],
            inplace=True,
        )

    merged.drop(columns=["_merge"], inplace=True)

    # Confirm source feature columns survived the merge unchanged.
    final_cnn_columns = [
        column for column in merged.columns
        if str(column).startswith(CNN_FEATURE_PREFIX)
    ]
    final_radiomics_columns = [
        column for column in merged.columns
        if str(column).startswith(RADIOMICS_FEATURE_PREFIX)
    ]

    if len(final_cnn_columns) != len(cnn_feature_columns):
        raise RuntimeError("CNN feature columns changed unexpectedly during merge.")
    if len(final_radiomics_columns) != len(radiomics_feature_columns):
        raise RuntimeError(
            "Radiomics feature columns changed unexpectedly during merge."
        )

    expected_total = (
        expected_cnn_features + expected_radiomics_features
        if expected_cnn_features is not None
        and expected_radiomics_features is not None
        else len(final_cnn_columns) + len(final_radiomics_columns)
    )

    if expected_records is not None and len(merged) != expected_records:
        raise ValueError(
            f"{partition}: expected {expected_records} matched records, "
            f"found {len(merged)}."
        )
    if len(merged) != len(cnn):
        raise RuntimeError(
            f"{partition}: merged row count differs from CNN input row count."
        )
    if not merged["filename"].astype(str).is_unique:
        raise RuntimeError("Merged output contains duplicate filenames.")

    audit = {
        "partition": partition,
        "matched_records": int(len(merged)),
        "merge_status": merge_counts,
        "cnn_features": int(len(final_cnn_columns)),
        "radiomics_features": int(len(final_radiomics_columns)),
        "combined_features": int(len(final_cnn_columns) + len(final_radiomics_columns)),
        "expected_combined_features": int(expected_total),
        "metadata_mismatches": metadata_mismatches,
        "side_values": {
            str(key): int(value)
            for key, value in merged["side"].value_counts(dropna=False).items()
        },
    }

    return merged, audit


def merge_feature_partitions(
    cnn_partitions: Mapping[str, pd.DataFrame],
    radiomics_partitions: Mapping[str, pd.DataFrame],
    *,
    expected_counts: Optional[Mapping[str, int]] = None,
    expected_cnn_features: Optional[int] = 512,
    expected_radiomics_features: Optional[int] = 102,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Merge train, validation, and test CNN/radiomics feature DataFrames."""
    missing_cnn = set(PARTITIONS).difference(cnn_partitions)
    missing_radiomics = set(PARTITIONS).difference(radiomics_partitions)
    if missing_cnn or missing_radiomics:
        raise ValueError(
            "Both input mappings must contain train, validation, and test. "
            f"Missing CNN: {sorted(missing_cnn)}; "
            f"missing radiomics: {sorted(missing_radiomics)}."
        )

    fused_partitions = {}
    audits = {}

    for partition in PARTITIONS:
        expected_records = (
            None if expected_counts is None
            else expected_counts.get(partition)
        )
        fused, audit = merge_cnn_radiomics_features(
            cnn_partitions[partition],
            radiomics_partitions[partition],
            partition,
            expected_cnn_features=expected_cnn_features,
            expected_radiomics_features=expected_radiomics_features,
            expected_records=expected_records,
        )
        fused_partitions[partition] = fused
        audits[partition] = audit

    return fused_partitions, audits
