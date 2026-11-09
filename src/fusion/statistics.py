"""Subject-level cluster bootstrap for CNN, radiomics, and hybrid models.

Converts the statistical comparison block from the original Colab workflow
into reusable functions.

The notebook's bootstrap cell is present in the source export, but the cell
that originally assembled ``bootstrap_eval_df`` is not. Therefore
``build_bootstrap_evaluation_table`` reconstructs that alignment explicitly:
- average CNN image probabilities within each group_id;
- align CNN, radiomics, and hybrid predictions by group_id;
- require identical subject IDs and ground-truth grades.

The bootstrap itself follows the exported notebook settings: 2,000 subject
cluster resamples, random seed 42, percentile 95% confidence intervals,
accuracy and macro-F1, plus paired Hybrid-minus-CNN and
Hybrid-minus-Radiomics differences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


MODELS = {
    "CNN": "cnn_prediction",
    "Radiomics": "radiomics_prediction",
    "Hybrid": "hybrid_prediction",
}
GRADE_LABELS = [1, 2, 3, 4]
PARTITIONS = ("train", "validation", "test")
DEFAULT_CNN_PROBABILITY_COLUMNS = tuple(
    f"probability_G{grade}" for grade in GRADE_LABELS
)
ALTERNATE_CNN_PROBABILITY_COLUMNS = tuple(
    f"G{grade}_probability" for grade in GRADE_LABELS
)


def _grade_to_integer(values: pd.Series, column_name: str) -> pd.Series:
    """Convert integer grades or labels such as G1 / Grade 1 to grades 1–4."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = numeric.copy()
    missing_mask = result.isna()

    if missing_mask.any():
        text_values = values.astype("string").str.strip().str.upper()
        extracted = text_values.str.extract(r"(?:GRADE\s*|G\s*)([1-4])", expand=False)
        extracted_numeric = pd.to_numeric(extracted, errors="coerce")
        result.loc[missing_mask] = extracted_numeric.loc[missing_mask]

    if result.isna().any() or not result.isin(GRADE_LABELS).all():
        bad = values.loc[
            result.isna() | ~result.isin(GRADE_LABELS)
        ].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{column_name} must contain grades 1–4 or names G1–G4. "
            f"Invalid values: {bad}"
        )

    if not (result == result.astype(int)).all():
        raise ValueError(f"{column_name} must contain integer grades.")
    return result.astype(int)


def _find_grade_column(dataframe: pd.DataFrame, preferred: Optional[str], role: str) -> str:
    """Resolve the actual-grade column without silently guessing unrelated fields."""
    candidates = [preferred] if preferred else []
    candidates.extend(["true_grade", "heckmatt_grade"])
    for candidate in candidates:
        if candidate and candidate in dataframe.columns:
            return candidate
    raise ValueError(
        f"Could not find the {role} grade column. Expected one of "
        f"{list(dict.fromkeys(c for c in candidates if c))}."
    )


def _find_cnn_probability_columns(
    dataframe: pd.DataFrame,
    requested: Optional[Sequence[str]] = None,
) -> list[str]:
    """Resolve the four CNN class probability columns."""
    if requested is not None:
        columns = list(requested)
        missing = set(columns).difference(dataframe.columns)
        if missing:
            raise ValueError(
                "Requested CNN probability columns are missing: "
                + ", ".join(sorted(missing))
            )
        if len(columns) != 4:
            raise ValueError("Exactly four CNN probability columns are required.")
        return columns

    for candidate in (
        list(DEFAULT_CNN_PROBABILITY_COLUMNS),
        list(ALTERNATE_CNN_PROBABILITY_COLUMNS),
    ):
        if set(candidate).issubset(dataframe.columns):
            return candidate

    raise ValueError(
        "CNN predictions must contain probability_G1..G4 or "
        "G1_probability..G4_probability columns."
    )


def _prepare_group_prediction_table(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    prediction_column: str,
    true_grade_column: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a group-level prediction table to shared comparison columns."""
    required = {"group_id", "subject_id", prediction_column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"{model_name} table is missing columns: {sorted(missing)}."
        )

    grade_column = _find_grade_column(
        dataframe,
        true_grade_column,
        f"{model_name} true",
    )
    result = dataframe[
        ["group_id", "subject_id", grade_column, prediction_column]
    ].copy()
    result = result.rename(
        columns={
            grade_column: "true_grade",
            prediction_column: prediction_column,
        }
    )

    if result[["group_id", "subject_id", "true_grade", prediction_column]].isna().any().any():
        raise ValueError(f"{model_name} table contains missing alignment values.")
    if result["group_id"].astype(str).duplicated().any():
        raise ValueError(f"{model_name} table must contain one row per group_id.")

    result["group_id"] = result["group_id"].astype(str)
    result["subject_id"] = result["subject_id"].astype(str)
    result["true_grade"] = _grade_to_integer(
        result["true_grade"], f"{model_name} true_grade"
    )
    result[prediction_column] = _grade_to_integer(
        result[prediction_column], f"{model_name} prediction"
    )
    return result


def build_bootstrap_evaluation_table(
    cnn_image_predictions: pd.DataFrame,
    radiomics_group_predictions: pd.DataFrame,
    hybrid_group_predictions: pd.DataFrame,
    *,
    cnn_probability_columns: Optional[Sequence[str]] = None,
    cnn_true_grade_column: Optional[str] = None,
    radiomics_true_grade_column: Optional[str] = None,
    hybrid_true_grade_column: Optional[str] = None,
    radiomics_prediction_column: str = "predicted_grade",
    hybrid_prediction_column: str = "predicted_grade",
    expected_groups: Optional[int] = 208,
    expected_subjects: Optional[int] = 16,
    probability_tolerance: float = 1e-5,
) -> pd.DataFrame:
    """Align CNN image, radiomics group, and hybrid group predictions.

    CNN probabilities are averaged within each group_id and converted to a
    grade using argmax, matching the project's mean-probability group
    aggregation. The three group-level tables are joined one-to-one by
    group_id. Every model must agree on subject_id and true_grade.

    The original bootstrap alignment cell's implementation is absent from
    the exported source text, so the alignment logic here is made explicit
    and validated rather than being presented as a verbatim cell conversion.
    """
    required_cnn = {"group_id", "subject_id"}
    missing_cnn = required_cnn.difference(cnn_image_predictions.columns)
    if missing_cnn:
        raise ValueError(
            "CNN image predictions are missing columns: "
            + ", ".join(sorted(missing_cnn))
        )
    if cnn_image_predictions.empty:
        raise ValueError("CNN image prediction table is empty.")
    if cnn_image_predictions["group_id"].isna().any():
        raise ValueError("CNN image predictions contain missing group_id values.")

    cnn_true_column = _find_grade_column(
        cnn_image_predictions,
        cnn_true_grade_column,
        "CNN true",
    )
    probability_columns = _find_cnn_probability_columns(
        cnn_image_predictions,
        cnn_probability_columns,
    )

    cnn_work = cnn_image_predictions[
        ["group_id", "subject_id", cnn_true_column, *probability_columns]
    ].copy()
    if cnn_work.isna().any().any():
        raise ValueError("CNN image predictions contain missing values.")

    cnn_work["group_id"] = cnn_work["group_id"].astype(str)
    cnn_work["subject_id"] = cnn_work["subject_id"].astype(str)
    cnn_work["true_grade"] = _grade_to_integer(
        cnn_work[cnn_true_column], "CNN true grade"
    )

    numeric_probabilities = cnn_work[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric_probabilities.isna().any().any():
        raise ValueError("CNN probabilities contain missing or non-numeric values.")
    probability_array = numeric_probabilities.to_numpy(dtype=float)
    if not np.isfinite(probability_array).all():
        raise ValueError("CNN probabilities contain non-finite values.")
    if (probability_array < 0).any() or (probability_array > 1).any():
        raise ValueError("CNN probabilities must be within [0, 1].")
    probability_sums = probability_array.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=probability_tolerance):
        raise ValueError("CNN class probabilities do not sum to one.")

    cnn_work.loc[:, probability_columns] = numeric_probabilities
    label_consistency = cnn_work.groupby("group_id")["true_grade"].nunique()
    subject_consistency = cnn_work.groupby("group_id")["subject_id"].nunique()
    if (label_consistency != 1).any():
        raise ValueError("CNN images within a group have inconsistent true grades.")
    if (subject_consistency != 1).any():
        raise ValueError("CNN images within a group have inconsistent subject IDs.")

    cnn_group = (
        cnn_work.groupby("group_id", sort=True)
        .agg(
            true_grade=("true_grade", "first"),
            subject_id=("subject_id", "first"),
            **{
                column: (column, "mean")
                for column in probability_columns
            },
        )
        .reset_index()
    )
    mean_probabilities = cnn_group[probability_columns].to_numpy(dtype=float)
    cnn_group["cnn_prediction"] = (
        np.argmax(mean_probabilities, axis=1) + 1
    )

    radiomics_group = _prepare_group_prediction_table(
        radiomics_group_predictions,
        model_name="Radiomics",
        prediction_column=radiomics_prediction_column,
        true_grade_column=radiomics_true_grade_column,
    ).rename(
        columns={
            "true_grade": "radiomics_true_grade",
            radiomics_prediction_column: "radiomics_prediction",
            "subject_id": "radiomics_subject_id",
        }
    )

    hybrid_group = _prepare_group_prediction_table(
        hybrid_group_predictions,
        model_name="Hybrid",
        prediction_column=hybrid_prediction_column,
        true_grade_column=hybrid_true_grade_column,
    ).rename(
        columns={
            "true_grade": "hybrid_true_grade",
            hybrid_prediction_column: "hybrid_prediction",
            "subject_id": "hybrid_subject_id",
        }
    )

    aligned = cnn_group[
        ["group_id", "subject_id", "true_grade", "cnn_prediction"]
    ].merge(
        radiomics_group[
            ["group_id", "radiomics_subject_id",
             "radiomics_true_grade", "radiomics_prediction"]
        ],
        on="group_id",
        how="outer",
        validate="one_to_one",
        indicator="_radiomics_merge",
    )

    if not aligned["_radiomics_merge"].eq("both").all():
        missing = aligned.loc[
            aligned["_radiomics_merge"] != "both", "group_id"
        ].head(10).tolist()
        raise ValueError(
            "CNN and radiomics group IDs do not align. "
            f"Examples: {missing}"
        )
    aligned = aligned.drop(columns="_radiomics_merge")

    radiomics_subject_mismatch = aligned["subject_id"].astype(str).ne(
        aligned["radiomics_subject_id"].astype(str)
    )
    if radiomics_subject_mismatch.any():
        examples = aligned.loc[
            radiomics_subject_mismatch,
            ["group_id", "subject_id", "radiomics_subject_id"],
        ].head(10).to_dict(orient="records")
        raise ValueError(
            f"Subject IDs differ between CNN and radiomics tables: {examples}"
        )

    aligned = aligned.merge(
        hybrid_group[
            ["group_id", "hybrid_subject_id",
             "hybrid_true_grade", "hybrid_prediction"]
        ],
        on="group_id",
        how="outer",
        validate="one_to_one",
        indicator="_hybrid_merge",
    )

    if not aligned["_hybrid_merge"].eq("both").all():
        missing = aligned.loc[
            aligned["_hybrid_merge"] != "both", "group_id"
        ].head(10).tolist()
        raise ValueError(
            "CNN/radiomics and hybrid group IDs do not align. "
            f"Examples: {missing}"
        )
    aligned = aligned.drop(columns="_hybrid_merge")

    subject_mismatch = aligned["subject_id"].astype(str).ne(
        aligned["hybrid_subject_id"].astype(str)
    )
    if subject_mismatch.any():
        examples = aligned.loc[
            subject_mismatch,
            ["group_id", "subject_id", "hybrid_subject_id"],
        ].head(10).to_dict(orient="records")
        raise ValueError(
            f"Subject IDs differ between CNN/radiomics and hybrid tables: {examples}"
        )

    grade_mismatch = (
        aligned["true_grade"].ne(aligned["radiomics_true_grade"])
        | aligned["true_grade"].ne(aligned["hybrid_true_grade"])
    )
    if grade_mismatch.any():
        examples = aligned.loc[
            grade_mismatch,
            [
                "group_id", "true_grade",
                "radiomics_true_grade", "hybrid_true_grade"
            ],
        ].head(10).to_dict(orient="records")
        raise ValueError(
            f"Ground-truth grade mismatch across model predictions: {examples}"
        )

    aligned = aligned.drop(
        columns=[
            "radiomics_subject_id",
            "hybrid_subject_id",
            "radiomics_true_grade",
            "hybrid_true_grade",
        ]
    )
    aligned = aligned.sort_values("group_id").reset_index(drop=True)

    if aligned["group_id"].duplicated().any():
        raise RuntimeError("Aligned evaluation table contains duplicate group IDs.")

    if expected_groups is not None and len(aligned) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} aligned groups, found {len(aligned)}."
        )
    subject_count = int(aligned["subject_id"].nunique())
    if expected_subjects is not None and subject_count != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} test subjects, found {subject_count}."
        )

    aligned["subject_id"] = aligned["subject_id"].astype(str)

    for model_column in ("cnn_prediction", "radiomics_prediction", "hybrid_prediction"):
        if not aligned[model_column].between(1, 4).all():
            raise ValueError(f"{model_column} contains grades outside 1–4.")

    return aligned[
        [
            "group_id",
            "subject_id",
            "true_grade",
            "cnn_prediction",
            "radiomics_prediction",
            "hybrid_prediction",
        ]
    ]


def _calculate_metrics(y_true: np.ndarray, y_prediction: np.ndarray) -> dict[str, float]:
    """Calculate the two metrics used in the source notebook."""
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_prediction,
                labels=GRADE_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }


def run_subject_cluster_bootstrap(
    aligned_predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> dict:
    """Run paired subject-level cluster bootstrap for all three models.

    Each replicate samples the unique subjects with replacement and includes
    every group belonging to each sampled subject. The same sampled indices
    are used for CNN, radiomics, and hybrid predictions, preserving pairing.

    Returns point estimates, model-level percentile confidence intervals,
    paired Hybrid-minus-CNN and Hybrid-minus-Radiomics differences, and
    methodology metadata.
    """
    required = {
        "group_id",
        "subject_id",
        "true_grade",
        "cnn_prediction",
        "radiomics_prediction",
        "hybrid_prediction",
    }
    missing = required.difference(aligned_predictions.columns)
    if missing:
        raise ValueError(
            "Aligned prediction table is missing columns: "
            + ", ".join(sorted(missing))
        )
    if aligned_predictions.empty:
        raise ValueError("Aligned prediction table is empty.")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")

    data = aligned_predictions.copy()
    if data[list(required)].isna().any().any():
        raise ValueError("Aligned predictions contain missing values.")
    if data["group_id"].astype(str).duplicated().any():
        raise ValueError("Aligned table must contain one row per group_id.")

    data["true_grade"] = _grade_to_integer(data["true_grade"], "true_grade")
    for column in ("cnn_prediction", "radiomics_prediction", "hybrid_prediction"):
        data[column] = _grade_to_integer(data[column], column)

    subjects = np.sort(data["subject_id"].astype(str).unique())
    if len(subjects) < 2:
        raise ValueError("At least two independent subjects are required.")

    y_true = data["true_grade"].to_numpy(dtype=int)
    predictions = {
        name: data[column].to_numpy(dtype=int)
        for name, column in MODELS.items()
    }

    subject_ids = data["subject_id"].astype(str).to_numpy()
    subject_indices = {
        subject: np.where(subject_ids == subject)[0]
        for subject in subjects
    }

    point_metrics = {
        name: _calculate_metrics(y_true, prediction)
        for name, prediction in predictions.items()
    }

    rng = np.random.default_rng(random_seed)
    model_samples = {
        name: {"accuracy": [], "macro_f1": []}
        for name in MODELS
    }
    paired_samples = {
        "Hybrid_minus_CNN": {"accuracy": [], "macro_f1": []},
        "Hybrid_minus_Radiomics": {"accuracy": [], "macro_f1": []},
    }

    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(
            subjects,
            size=len(subjects),
            replace=True,
        )
        indices = np.concatenate([
            subject_indices[subject]
            for subject in sampled_subjects
        ])
        y_boot = y_true[indices]

        replicate_metrics = {}
        for model_name, prediction in predictions.items():
            metrics = _calculate_metrics(y_boot, prediction[indices])
            replicate_metrics[model_name] = metrics
            for metric in ("accuracy", "macro_f1"):
                model_samples[model_name][metric].append(metrics[metric])

        for baseline in ("CNN", "Radiomics"):
            comparison = f"Hybrid_minus_{baseline}"
            for metric in ("accuracy", "macro_f1"):
                difference = (
                    replicate_metrics["Hybrid"][metric]
                    - replicate_metrics[baseline][metric]
                )
                paired_samples[comparison][metric].append(float(difference))

    summary_rows = []
    for model_name in MODELS:
        for metric in ("accuracy", "macro_f1"):
            values = np.asarray(model_samples[model_name][metric], dtype=float)
            lower, upper = np.percentile(values, [2.5, 97.5])
            summary_rows.append({
                "Model": model_name,
                "Metric": metric,
                "Test_Estimate": point_metrics[model_name][metric],
                "CI_95_Lower": float(lower),
                "CI_95_Upper": float(upper),
            })

    difference_rows = []
    for comparison, metrics in paired_samples.items():
        for metric, samples in metrics.items():
            values = np.asarray(samples, dtype=float)
            lower, upper = np.percentile(values, [2.5, 97.5])
            difference_rows.append({
                "Comparison": comparison,
                "Metric": metric,
                "Mean_Bootstrap_Difference": float(values.mean()),
                "CI_95_Lower": float(lower),
                "CI_95_Upper": float(upper),
            })

    grade_coverage = {
        f"G{grade}": int(
            data.loc[data["true_grade"] == grade, "subject_id"].nunique()
        )
        for grade in GRADE_LABELS
    }

    metadata = {
        "analysis": "Subject-level cluster bootstrap",
        "bootstrap_repetitions": int(n_bootstrap),
        "random_seed": int(random_seed),
        "resampling_unit": "Subject",
        "independent_test_subjects": int(len(subjects)),
        "test_groups": int(len(data)),
        "grade_coverage_subjects": grade_coverage,
        "confidence_interval": "Percentile 95% CI",
        "metrics": ["accuracy", "macro_f1"],
        "models": [
            "ResNet18 CNN",
            "Radiomics XGBoost",
            "CNN-Radiomics Hybrid XGBoost",
        ],
        "paired_comparisons": [
            "Hybrid_minus_CNN",
            "Hybrid_minus_Radiomics",
        ],
        "interpretation": (
            "Interpret paired confidence intervals against zero. If a 95% "
            "interval includes zero, this bootstrap analysis does not establish "
            "a clear difference for that metric."
        ),
        "limitations": [
            f"Only {len(subjects)} independent test subjects are represented.",
            "Grade-specific subject coverage may be sparse.",
            "Cluster bootstrap estimates may be unstable with few subjects.",
        ],
    }

    return {
        "aligned_predictions": data,
        "point_metrics": point_metrics,
        "model_confidence_intervals": pd.DataFrame(summary_rows),
        "paired_model_differences": pd.DataFrame(difference_rows),
        "metadata": metadata,
    }


def save_subject_bootstrap_results(
    bootstrap_result: dict,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save the model CIs, paired comparisons, alignment table, and metadata."""
    required = {
        "aligned_predictions",
        "model_confidence_intervals",
        "paired_model_differences",
        "metadata",
    }
    missing = required.difference(bootstrap_result)
    if missing:
        raise ValueError(
            "Bootstrap result is missing keys: "
            + ", ".join(sorted(missing))
        )

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    paths = {
        "aligned_predictions": destination / "Subject_Bootstrap_Aligned_Predictions.csv",
        "model_confidence_intervals": destination / "Subject_Bootstrap_Model_Confidence_Intervals.csv",
        "paired_model_differences": destination / "Subject_Bootstrap_Paired_Model_Differences.csv",
        "methodology": destination / "Subject_Bootstrap_Methodology_and_Findings.json",
    }

    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing bootstrap artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Pass overwrite=True to replace them."
        )

    bootstrap_result["aligned_predictions"].to_csv(
        paths["aligned_predictions"], index=False
    )
    bootstrap_result["model_confidence_intervals"].to_csv(
        paths["model_confidence_intervals"], index=False
    )
    bootstrap_result["paired_model_differences"].to_csv(
        paths["paired_model_differences"], index=False
    )
    paths["methodology"].write_text(
        json.dumps(bootstrap_result["metadata"], indent=4),
        encoding="utf-8",
    )

    return paths
