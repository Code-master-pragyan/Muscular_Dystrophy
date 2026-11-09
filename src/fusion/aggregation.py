"""Group-level aggregation for the Dataset B CNN-radiomics hybrid pipeline.

Converts the original Colab aggregation/save block into reusable functions.
Averages all 512 CNN and 102 radiomics image-level features within each
subject–muscle–side group_id, preserves one metadata row per group, validates
the expected partition sizes, and optionally saves group CSVs and config JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")
DEFAULT_EXPECTED_GROUPS = {
    "train": 1029,
    "validation": 231,
    "test": 208,
}
METADATA_COLUMNS = (
    "subject_id",
    "muscle_code",
    "side_code",
    "side",
    "heckmatt_grade",
    "partition",
)
CNN_FEATURE_PREFIX = "cnn_feature_"
RADIOMICS_FEATURE_PREFIX = "original_"


def _get_feature_columns(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return CNN and radiomics feature columns in stable column order."""
    cnn_columns = [
        str(column) for column in dataframe.columns
        if str(column).startswith(CNN_FEATURE_PREFIX)
    ]
    radiomics_columns = [
        str(column) for column in dataframe.columns
        if str(column).startswith(RADIOMICS_FEATURE_PREFIX)
    ]
    if not cnn_columns:
        raise ValueError("No CNN columns prefixed with 'cnn_feature_' found.")
    if not radiomics_columns:
        raise ValueError("No radiomics columns prefixed with 'original_' found.")
    return cnn_columns, radiomics_columns


def aggregate_fused_features_by_group(
    fused_df: pd.DataFrame,
    partition: str,
    *,
    expected_groups: Optional[int] = None,
    expected_cnn_features: Optional[int] = 512,
    expected_radiomics_features: Optional[int] = 102,
) -> tuple[pd.DataFrame, dict]:
    """Average all fused image-level feature columns within each group_id."""
    partition = str(partition).strip().lower()
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}.")
    if not isinstance(fused_df, pd.DataFrame):
        raise TypeError("fused_df must be a pandas DataFrame.")
    if fused_df.empty:
        raise ValueError("fused_df is empty.")

    required = {"group_id", *METADATA_COLUMNS}
    missing = required.difference(fused_df.columns)
    if missing:
        raise ValueError(
            "Fused DataFrame is missing columns: "
            + ", ".join(sorted(missing))
        )
    if fused_df["group_id"].isna().any():
        raise ValueError("Fused DataFrame contains missing group_id values.")
    if fused_df[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Fused DataFrame contains missing group metadata.")

    partitions_found = set(
        fused_df["partition"].astype(str).str.strip().str.lower()
    )
    if partitions_found != {partition}:
        raise ValueError(
            f"Expected only partition {partition!r}; found {sorted(partitions_found)}."
        )

    cnn_columns, radiomics_columns = _get_feature_columns(fused_df)
    if expected_cnn_features is not None and len(cnn_columns) != expected_cnn_features:
        raise ValueError(
            f"Expected {expected_cnn_features} CNN features, found {len(cnn_columns)}."
        )
    if (
        expected_radiomics_features is not None
        and len(radiomics_columns) != expected_radiomics_features
    ):
        raise ValueError(
            f"Expected {expected_radiomics_features} radiomics features, "
            f"found {len(radiomics_columns)}."
        )

    feature_columns = cnn_columns + radiomics_columns
    features = fused_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing/non-numeric feature values: {bad[:10]}.")
    values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Fused features contain NaN or infinite values.")

    working = fused_df[["group_id", *METADATA_COLUMNS]].copy()
    working.loc[:, feature_columns] = features

    # The source notebook checked grade consistency and retained first metadata.
    # Validate all group-identifying metadata before preserving one row.
    consistency_columns = list(METADATA_COLUMNS)
    distinct = (
        working.groupby("group_id", sort=False)[consistency_columns]
        .nunique(dropna=False)
    )
    inconsistent = distinct.gt(1)
    if inconsistent.any().any():
        issues = [
            (str(group_id), str(column))
            for group_id, row in inconsistent.iterrows()
            for column, failed in row.items()
            if failed
        ]
        raise ValueError(
            "Inconsistent metadata within group_id values. "
            f"Examples: {issues[:10]}"
        )

    grouped_features = (
        working.groupby("group_id", sort=True)[feature_columns]
        .mean()
        .reset_index()
    )
    grouped_metadata = (
        working.groupby("group_id", sort=True)[list(METADATA_COLUMNS)]
        .first()
        .reset_index()
    )
    group_df = grouped_metadata.merge(
        grouped_features,
        on="group_id",
        validate="one_to_one",
    )

    if group_df["group_id"].duplicated().any():
        raise RuntimeError("Aggregated output contains duplicate group_id values.")
    if group_df["heckmatt_grade"].isna().any():
        raise RuntimeError("Aggregated output contains missing Heckmatt grades.")

    aggregated = group_df[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(aggregated).all():
        raise RuntimeError("Aggregated features contain non-finite values.")
    if expected_groups is not None and len(group_df) != expected_groups:
        raise ValueError(
            f"{partition}: expected {expected_groups} groups, found {len(group_df)}."
        )

    grade_distribution = {
        int(grade): int(count)
        for grade, count in (
            group_df["heckmatt_grade"].astype(int).value_counts().sort_index().items()
        )
    }
    audit = {
        "partition": partition,
        "image_records": int(len(fused_df)),
        "aggregated_groups": int(len(group_df)),
        "cnn_features": int(len(cnn_columns)),
        "radiomics_features": int(len(radiomics_columns)),
        "total_features": int(len(feature_columns)),
        "expected_groups": expected_groups,
        "grade_distribution": grade_distribution,
        "all_features_finite": bool(np.isfinite(aggregated).all()),
    }
    return group_df, audit


def aggregate_fusion_partitions(
    fused_partitions: Mapping[str, pd.DataFrame],
    *,
    expected_groups: Optional[Mapping[str, int]] = DEFAULT_EXPECTED_GROUPS,
    expected_cnn_features: Optional[int] = 512,
    expected_radiomics_features: Optional[int] = 102,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Aggregate train, validation, and test fused tables independently."""
    missing = set(PARTITIONS).difference(fused_partitions)
    if missing:
        raise ValueError(
            "fused_partitions must contain train, validation, and test; "
            f"missing: {sorted(missing)}."
        )

    tables = {}
    audits = {}
    for partition in PARTITIONS:
        expected = None if expected_groups is None else expected_groups.get(partition)
        tables[partition], audits[partition] = aggregate_fused_features_by_group(
            fused_partitions[partition],
            partition,
            expected_groups=expected,
            expected_cnn_features=expected_cnn_features,
            expected_radiomics_features=expected_radiomics_features,
        )
    return tables, audits


def save_group_fusion_datasets(
    group_tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    random_seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save group-level fusion CSVs and Dataset_B_Hybrid_Fusion_Config.json."""
    missing = set(PARTITIONS).difference(group_tables)
    if missing:
        raise ValueError(
            "group_tables must contain train, validation, and test; "
            f"missing: {sorted(missing)}."
        )

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    file_paths = {
        partition: destination
        / f"Dataset_B_Hybrid_Fusion_{partition.capitalize()}_Groups.csv"
        for partition in PARTITIONS
    }
    config_path = destination / "Dataset_B_Hybrid_Fusion_Config.json"
    targets = [*file_paths.values(), config_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Pass overwrite=True to replace them."
        )

    counts = {}
    feature_dimensions = None
    for partition in PARTITIONS:
        table = group_tables[partition]
        if table.empty:
            raise ValueError(f"{partition} group table is empty.")
        if "group_id" not in table or table["group_id"].duplicated().any():
            raise ValueError(f"{partition} must have unique group_id values.")
        if "heckmatt_grade" not in table or table["heckmatt_grade"].isna().any():
            raise ValueError(f"{partition} must have non-missing Heckmatt grades.")

        cnn_columns, radiomics_columns = _get_feature_columns(table)
        dimensions = (len(cnn_columns), len(radiomics_columns))
        if feature_dimensions is None:
            feature_dimensions = dimensions
        elif dimensions != feature_dimensions:
            raise ValueError("Feature dimensions differ across partitions.")
        counts[partition] = {"groups": int(len(table))}

    for partition in PARTITIONS:
        group_tables[partition].to_csv(file_paths[partition], index=False)

    cnn_count, radiomics_count = feature_dimensions
    config = {
        "experiment": "CNN-Radiomics Feature Concatenation",
        "cnn_architecture": "ResNet18",
        "cnn_features": cnn_count,
        "radiomics_features": radiomics_count,
        "total_features": cnn_count + radiomics_count,
        "aggregation": "Mean aggregation by group_id",
        "group_definition": "Subject + Muscle + Side",
        "classification_classes": ["G1", "G2", "G3", "G4"],
        "random_seed": int(random_seed),
        "partitions": counts,
    }
    config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")
    return {**file_paths, "config": config_path}
