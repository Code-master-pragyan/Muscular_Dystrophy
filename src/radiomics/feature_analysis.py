
"""Exploratory quality, redundancy, and grade-discrimination analysis.

Converts the radiomics feature-quality and grade-discrimination audit blocks
from the original Colab workflow into reusable functions.

The analysis:
- removes exact duplicate feature columns from the analysis copy;
- reports missing, infinite, constant, near-constant, and duplicate features;
- reports highly correlated feature pairs;
- aggregates image-level features to subject–muscle–side group means;
- applies Kruskal–Wallis tests across Heckmatt grades 1–4;
- adjusts p-values using Benjamini–Hochberg FDR;
- compares the top training features descriptively across validation/test.

This is an exploratory audit, not a model feature-selection procedure.
Validation/test results must not be used to choose model features or tune
the final model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests


PARTITIONS = ("train", "validation", "test")
GRADES = (1, 2, 3, 4)

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


def _select_feature_columns(
    dataframe: pd.DataFrame,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> list[str]:
    """Select numeric radiomics columns, excluding available metadata."""
    metadata = set(metadata_columns)

    candidates = [
        column
        for column in dataframe.columns
        if column not in metadata
    ]

    numeric = [
        column
        for column in candidates
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if not numeric:
        raise ValueError("No numeric feature columns were found.")

    return numeric


def _drop_exact_duplicate_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove identical feature columns, preserving the first occurrence."""
    duplicate_mask = features.T.duplicated(keep="first")

    duplicate_names = (
        features.columns[duplicate_mask].astype(str).tolist()
    )

    return features.loc[:, ~duplicate_mask].copy(), duplicate_names


def audit_radiomics_feature_quality(
    feature_table: pd.DataFrame,
    *,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    near_constant_unique_ratio: float = 0.01,
) -> dict:
    """Summarize quality and redundancy of one image-level feature table."""

    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")

    if feature_table.empty:
        raise ValueError("feature_table is empty.")

    if not 0.0 <= near_constant_unique_ratio <= 1.0:
        raise ValueError(
            "near_constant_unique_ratio must be in [0, 1]."
        )

    feature_columns = _select_feature_columns(
        feature_table,
        metadata_columns,
    )

    numeric = feature_table[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_counts = numeric.isna().sum()

    numeric_array = numeric.to_numpy(dtype=np.float64)

    infinite_counts = pd.Series(
        np.isinf(numeric_array).sum(axis=0),
        index=numeric.columns,
    )

    finite_numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    variance = finite_numeric.var(
        axis=0,
        skipna=True,
        ddof=1,
    )

    constant_features = [
        str(column)
        for column in finite_numeric.columns
        if finite_numeric[column].dropna().nunique() <= 1
    ]

    near_constant_features = []
    unique_ratios = {}

    for column in finite_numeric.columns:
        valid = finite_numeric[column].dropna()

        ratio = (
            float(valid.nunique() / len(valid))
            if len(valid)
            else float("nan")
        )

        unique_ratios[str(column)] = ratio

        if (
            len(valid) > 0
            and valid.nunique() > 1
            and ratio <= near_constant_unique_ratio
        ):
            near_constant_features.append(str(column))

    deduplicated, duplicate_columns = (
        _drop_exact_duplicate_features(numeric)
    )

    return {
        "records": int(len(feature_table)),

        "metadata_columns": [
            column
            for column in metadata_columns
            if column in feature_table.columns
        ],

        "feature_count": int(len(feature_columns)),

        "missing_values_by_feature": {
            str(column): int(count)
            for column, count in missing_counts.items()
        },

        "missing_value_total": int(missing_counts.sum()),

        "infinite_values_by_feature": {
            str(column): int(count)
            for column, count in infinite_counts.items()
        },

        "infinite_value_total": int(infinite_counts.sum()),

        "constant_features": constant_features,

        "near_constant_features": near_constant_features,

        "near_constant_unique_ratio_threshold": float(
            near_constant_unique_ratio
        ),

        "exact_duplicate_features": duplicate_columns,

        "unique_features_after_duplicate_removal": int(
            deduplicated.shape[1]
        ),

        "feature_variance": {
            str(column): (
                float(value) if pd.notna(value) else None
            )
            for column, value in variance.items()
        },

        "unique_value_ratio": unique_ratios,
    }


def find_highly_correlated_features(
    feature_table: pd.DataFrame,
    *,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    absolute_correlation_threshold: float = 0.999,
    drop_exact_duplicates: bool = True,
) -> pd.DataFrame:
    """Return feature pairs whose absolute Pearson correlation meets threshold."""

    if not -1.0 <= absolute_correlation_threshold <= 1.0:
        raise ValueError(
            "absolute_correlation_threshold must be between -1 and 1."
        )

    feature_columns = _select_feature_columns(
        feature_table,
        metadata_columns,
    )

    numeric = feature_table[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if drop_exact_duplicates:
        numeric, _ = _drop_exact_duplicate_features(numeric)

    correlation = numeric.corr(method="pearson")

    rows = []
    columns = list(correlation.columns)

    for left_index, left_name in enumerate(columns):
        for right_name in columns[left_index + 1:]:

            value = correlation.loc[left_name, right_name]

            if (
                pd.notna(value)
                and abs(float(value)) >= absolute_correlation_threshold
            ):
                rows.append({
                    "feature_1": str(left_name),
                    "feature_2": str(right_name),
                    "correlation": float(value),
                    "absolute_correlation": abs(float(value)),
                })

    result = pd.DataFrame(
        rows,
        columns=[
            "feature_1",
            "feature_2",
            "correlation",
            "absolute_correlation",
        ],
    )

    if not result.empty:
        result = result.sort_values(
            "absolute_correlation",
            ascending=False,
        ).reset_index(drop=True)

    return result


def _prepare_group_level_features(
    feature_table: pd.DataFrame,
    *,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Clean features and aggregate rows to group means with label checks."""

    required = {
        "group_id",
        "heckmatt_grade",
    }

    missing = required.difference(feature_table.columns)

    if missing:
        raise ValueError(
            "Feature table is missing columns: "
            + ", ".join(sorted(missing))
        )

    if feature_table[
        ["group_id", "heckmatt_grade"]
    ].isna().any().any():
        raise ValueError(
            "group_id and Heckmatt grade cannot contain missing values."
        )

    grades = pd.to_numeric(
        feature_table["heckmatt_grade"],
        errors="coerce",
    )

    if grades.isna().any() or not grades.isin(GRADES).all():
        raise ValueError(
            "Heckmatt grades must be integers from 1 through 4."
        )

    if not (grades == grades.astype(int)).all():
        raise ValueError("Heckmatt grades must be integer values.")

    feature_columns = _select_feature_columns(
        feature_table,
        metadata_columns,
    )

    features = feature_table[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    features, duplicate_features = (
        _drop_exact_duplicate_features(features)
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    work = pd.concat(
        [
            feature_table[["group_id"]].reset_index(drop=True),

            grades.astype(int)
            .rename("heckmatt_grade")
            .reset_index(drop=True),

            features.reset_index(drop=True),
        ],
        axis=1,
    )

    label_counts = (
        work.groupby("group_id")["heckmatt_grade"]
        .nunique()
    )

    inconsistent = label_counts[label_counts != 1]

    if not inconsistent.empty:
        raise ValueError(
            "Groups contain inconsistent Heckmatt grades: "
            f"{inconsistent.index.astype(str).tolist()[:10]}"
        )

    grouped = (
        work.groupby(
            ["group_id", "heckmatt_grade"],
            as_index=False,
            sort=True,
        )
        .mean(numeric_only=True)
    )

    return (
        grouped,
        list(features.columns),
        duplicate_features,
    )


def analyse_radiomics_grade_discrimination(
    feature_table: pd.DataFrame,
    partition_name: str,
    *,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run per-feature Kruskal–Wallis tests across group-level grades 1–4.

    Rank effect size follows the notebook formula:

        rank_effect_size = Kruskal statistic / (N_groups - 1)

    Features are tested only when each grade has at least two group-level
    observations. P-values are corrected using Benjamini–Hochberg FDR.
    """

    name = str(partition_name).strip().lower()

    if name not in PARTITIONS:
        raise ValueError(
            f"partition_name must be one of {PARTITIONS}."
        )

    grouped, feature_columns, duplicate_features = (
        _prepare_group_level_features(
            feature_table,
            metadata_columns=metadata_columns,
        )
    )

    result_rows = []
    skipped_features = {}

    for feature in feature_columns:

        grade_arrays = []
        insufficient_grades = []

        for grade in GRADES:

            values = grouped.loc[
                grouped["heckmatt_grade"] == grade,
                feature,
            ].dropna().to_numpy(dtype=float)

            if len(values) < 2:
                insufficient_grades.append(grade)

            grade_arrays.append(values)

        if insufficient_grades:
            skipped_features[str(feature)] = {
                "reason": (
                    "Fewer than two groups in one or more grades"
                ),
                "grades": insufficient_grades,
            }
            continue

        combined = np.concatenate(grade_arrays)

        if not np.isfinite(combined).all():
            skipped_features[str(feature)] = {
                "reason": "Non-finite group-level values"
            }
            continue

        try:
            statistic, p_value = kruskal(*grade_arrays)

        except ValueError as exc:
            skipped_features[str(feature)] = {
                "reason": str(exc)
            }
            continue

        if not np.isfinite(statistic) or not np.isfinite(p_value):
            skipped_features[str(feature)] = {
                "reason": (
                    "Kruskal–Wallis returned a non-finite result"
                )
            }
            continue

        n_groups_tested = int(
            sum(len(values) for values in grade_arrays)
        )

        effect_size = (
            float(statistic / (n_groups_tested - 1))
            if n_groups_tested > 1
            else float("nan")
        )

        result_rows.append({
            "feature": str(feature),
            "kruskal_statistic": float(statistic),
            "p_value": float(p_value),
            "rank_effect_size": effect_size,
            "groups_tested": n_groups_tested,
        })

    results = pd.DataFrame(
        result_rows,
        columns=[
            "feature",
            "kruskal_statistic",
            "p_value",
            "rank_effect_size",
            "groups_tested",
        ],
    )

    if results.empty:
        raise ValueError(
            "No feature had at least two group-level observations "
            "in every grade."
        )

    results["adjusted_p_value"] = multipletests(
        results["p_value"].to_numpy(dtype=float),
        method="fdr_bh",
    )[1]

    results = results.sort_values(
        ["adjusted_p_value", "p_value"],
        ascending=True,
    ).reset_index(drop=True)

    summary = {
        "partition": name,

        "image_records": int(len(feature_table)),

        "group_records": int(len(grouped)),

        "features_before_duplicate_removal": int(
            len(feature_columns) + len(duplicate_features)
        ),

        "duplicate_features_removed": duplicate_features,

        "features_tested": int(len(results)),

        "features_skipped": skipped_features,

        "significant_features_fdr_0_05": int(
            (results["adjusted_p_value"] < 0.05).sum()
        ),

        "group_counts_by_grade": {
            int(grade): int(
                (grouped["heckmatt_grade"] == grade).sum()
            )
            for grade in GRADES
        },

        "statistical_test": (
            "Kruskal-Wallis H test across grades 1-4"
        ),

        "multiple_testing_correction": (
            "Benjamini-Hochberg FDR"
        ),

        "rank_effect_size_formula": (
            "H / (N_groups_tested - 1)"
        ),
    }

    return grouped, results, summary


def compare_top_training_features_across_partitions(
    partition_results: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Compare top training-discriminating features across train/val/test.

    Feature ranking is derived only from the training results. Validation
    and test statistics are attached as a descriptive generalization audit.
    """

    missing = set(PARTITIONS).difference(partition_results)

    if missing:
        raise ValueError(
            "partition_results must contain train, validation, and test; "
            f"missing: {sorted(missing)}."
        )

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    train_results = partition_results["train"]

    required = {
        "feature",
        "adjusted_p_value",
        "rank_effect_size",
    }

    missing_columns = required.difference(train_results.columns)

    if missing_columns:
        raise ValueError(
            "Training results are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    top_features = (
        train_results.sort_values("adjusted_p_value")
        .head(top_n)["feature"]
        .astype(str)
        .tolist()
    )

    rows = []

    for feature in top_features:

        row = {"feature": feature}

        for partition in PARTITIONS:

            result = partition_results[partition]

            missing = required.difference(result.columns)

            if missing:
                raise ValueError(
                    f"{partition} results are missing columns: "
                    + ", ".join(sorted(missing))
                )

            match = result.loc[
                result["feature"].astype(str) == feature
            ]

            if not match.empty:

                row[f"{partition}_adjusted_p"] = float(
                    match.iloc[0]["adjusted_p_value"]
                )

                row[f"{partition}_effect_size"] = float(
                    match.iloc[0]["rank_effect_size"]
                )

            else:

                row[f"{partition}_adjusted_p"] = np.nan
                row[f"{partition}_effect_size"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def analyse_radiomics_partitions(
    feature_tables: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 10,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> dict:
    """Run exploratory grade-discrimination analysis on all three splits."""

    missing = set(PARTITIONS).difference(feature_tables)

    if missing:
        raise ValueError(
            "feature_tables must contain train, validation, and test; "
            f"missing: {sorted(missing)}."
        )

    grouped_tables = {}
    results_tables = {}
    summaries = {}

    for partition in PARTITIONS:

        grouped, results, summary = (
            analyse_radiomics_grade_discrimination(
                feature_tables[partition],
                partition,
                metadata_columns=metadata_columns,
            )
        )

        grouped_tables[partition] = grouped
        results_tables[partition] = results
        summaries[partition] = summary

    comparison = compare_top_training_features_across_partitions(
        results_tables,
        top_n=top_n,
    )

    return {
        "grouped_tables": grouped_tables,
        "feature_results": results_tables,
        "summaries": summaries,
        "top_training_feature_comparison": comparison,
    }


def save_radiomics_feature_analysis(
    analysis: dict,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save per-partition results, feature comparisons, and summaries."""

    required = {
        "grouped_tables",
        "feature_results",
        "summaries",
        "top_training_feature_comparison",
    }

    missing = required.difference(analysis)

    if missing:
        raise ValueError(
            "Analysis result is missing keys: "
            + ", ".join(sorted(missing))
        )

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    paths = {}

    for partition in PARTITIONS:

        paths[f"{partition}_grouped"] = (
            destination
            / f"Radiomics_{partition.capitalize()}_Group_Features.csv"
        )

        paths[f"{partition}_feature_tests"] = (
            destination
            / f"Radiomics_{partition.capitalize()}_Grade_Discrimination.csv"
        )

    paths["top_training_feature_comparison"] = (
        destination
        / "Radiomics_Top_Training_Feature_Generalisation.csv"
    )

    paths["summary"] = (
        destination
        / "Radiomics_Feature_Analysis_Summary.json"
    )

    existing = [
        path for path in paths.values()
        if path.exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing radiomics analysis files: "
            + ", ".join(str(path) for path in existing)
            + ". Pass overwrite=True to replace them."
        )

    for partition in PARTITIONS:

        analysis["grouped_tables"][partition].to_csv(
            paths[f"{partition}_grouped"],
            index=False,
        )

        analysis["feature_results"][partition].to_csv(
            paths[f"{partition}_feature_tests"],
            index=False,
        )

    analysis["top_training_feature_comparison"].to_csv(
        paths["top_training_feature_comparison"],
        index=False,
    )

    paths["summary"].write_text(
        json.dumps(analysis["summaries"], indent=4),
        encoding="utf-8",
    )

    return paths
