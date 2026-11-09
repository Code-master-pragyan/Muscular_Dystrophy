
"""Generate the three final research figures for Dataset B hybrid fusion.

Converts the final figure-generation block from the original Colab workflow.
The figures are built from reusable outputs produced by:
- src.fusion.statistics (bootstrap confidence intervals and aligned predictions)
- src.fusion.explainability (SHAP modality contribution)

Output figures:
1. Figure_1_Model_Performance_Confidence_Intervals.png
2. Figure_2_Group_Level_Confusion_Matrices.png
3. Figure_3_SHAP_Modality_Contribution.png

Matplotlib is imported lazily so the rest of the fusion package does not
require plotting support unless figure generation is requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


MODEL_ORDER = ("CNN", "Radiomics", "Hybrid")

PREDICTION_COLUMNS = {
    "CNN": "cnn_prediction",
    "Radiomics": "radiomics_prediction",
    "Hybrid": "hybrid_prediction",
}

GRADE_LABELS = (1, 2, 3, 4)
GRADE_TICKS = ("G1", "G2", "G3", "G4")

FIGURE_FILENAMES = {
    "performance": "Figure_1_Model_Performance_Confidence_Intervals.png",
    "confusion_matrices": "Figure_2_Group_Level_Confusion_Matrices.png",
    "shap_modality": "Figure_3_SHAP_Modality_Contribution.png",
}


def _load_matplotlib():
    """Import pyplot lazily and select a non-interactive backend."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate research figures. "
            "Install it with: pip install matplotlib"
        ) from exc

    return plt


def _validate_bootstrap_summary(summary: pd.DataFrame) -> None:
    required = {
        "Model",
        "Metric",
        "Test_Estimate",
        "CI_95_Lower",
        "CI_95_Upper",
    }

    missing = required.difference(summary.columns)

    if missing:
        raise ValueError(
            "Bootstrap summary is missing columns: "
            + ", ".join(sorted(missing))
        )

    if summary.empty:
        raise ValueError("Bootstrap summary is empty.")

    selected = summary[
        summary["Model"].isin(MODEL_ORDER)
        & summary["Metric"].isin(["accuracy", "macro_f1"])
    ]

    expected_pairs = {
        (model_name, metric_name)
        for model_name in MODEL_ORDER
        for metric_name in ("accuracy", "macro_f1")
    }

    actual_pairs = set(
        zip(selected["Model"], selected["Metric"])
    )

    if actual_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - actual_pairs)

        raise ValueError(
            f"Bootstrap summary lacks model/metric entries: {missing_pairs}"
        )

    numeric_columns = [
        "Test_Estimate",
        "CI_95_Lower",
        "CI_95_Upper",
    ]

    values = selected[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if values.isna().any().any() or not np.isfinite(
        values.to_numpy()
    ).all():
        raise ValueError(
            "Bootstrap summary contains missing/non-finite metrics."
        )

    invalid_ci = (
        (values["CI_95_Lower"] > values["Test_Estimate"])
        | (values["Test_Estimate"] > values["CI_95_Upper"])
    )

    if invalid_ci.any():
        raise ValueError(
            "Bootstrap confidence intervals do not contain estimates."
        )


def _validate_aligned_predictions(aligned: pd.DataFrame) -> None:
    required = {
        "true_grade",
        "cnn_prediction",
        "radiomics_prediction",
        "hybrid_prediction",
    }

    missing = required.difference(aligned.columns)

    if missing:
        raise ValueError(
            "Aligned predictions are missing columns: "
            + ", ".join(sorted(missing))
        )

    if aligned.empty:
        raise ValueError("Aligned prediction table is empty.")

    values = aligned[
        [
            "true_grade",
            "cnn_prediction",
            "radiomics_prediction",
            "hybrid_prediction",
        ]
    ].apply(pd.to_numeric, errors="coerce")

    if values.isna().any().any():
        raise ValueError(
            "Aligned predictions contain missing/non-numeric grades."
        )

    if not values.apply(
        lambda column: column.isin(GRADE_LABELS)
    ).all().all():
        raise ValueError(
            "Aligned predictions must contain grades 1 through 4."
        )


def _validate_modality_importance(
    modality: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "modality",
        "contribution_percent",
    }

    missing = required.difference(modality.columns)

    if missing:
        raise ValueError(
            "Modality contribution table is missing columns: "
            + ", ".join(sorted(missing))
        )

    result = modality.copy()
    result["modality"] = result["modality"].astype(str)

    expected = {"CNN", "Radiomics"}
    found = set(result["modality"])

    if found != expected:
        raise ValueError(
            "Modality contribution table must contain CNN and Radiomics; "
            f"found {sorted(found)}."
        )

    result["contribution_percent"] = pd.to_numeric(
        result["contribution_percent"],
        errors="coerce",
    )

    if (
        result["contribution_percent"].isna().any()
        or not np.isfinite(result["contribution_percent"]).all()
    ):
        raise ValueError(
            "Modality contributions must be finite numeric values."
        )

    if (
        (result["contribution_percent"] < 0).any()
        or (result["contribution_percent"] > 100).any()
    ):
        raise ValueError(
            "Modality contributions must lie between 0 and 100."
        )

    total = float(
        result["contribution_percent"].sum()
    )

    if total > 0 and not np.isclose(total, 100.0, atol=1e-3):
        raise ValueError(
            f"Non-zero modality contributions must sum to 100%; "
            f"found {total:.4f}%."
        )

    return (
        result.set_index("modality")
        .loc[["CNN", "Radiomics"]]
        .reset_index()
    )


def generate_final_hybrid_figures(
    model_confidence_intervals: pd.DataFrame,
    aligned_predictions: pd.DataFrame,
    modality_contribution: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    class_labels: Sequence[str] = GRADE_TICKS,
) -> dict[str, Path]:
    """Generate and save the three notebook figures.

    Parameters
    ----------
    model_confidence_intervals:
        model_confidence_intervals DataFrame from
        run_subject_cluster_bootstrap.

    aligned_predictions:
        Aligned group-level predictions DataFrame with true_grade and
        prediction columns for each model.

    modality_contribution:
        modality_contribution DataFrame from
        calculate_hybrid_shap_explainability.

    output_dir:
        Destination directory for PNG figures.

    dpi:
        Figure resolution. Defaults to 300.

    class_labels:
        Four displayed class tick labels, in grade order.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by performance, confusion_matrices, and shap_modality.
    """

    _validate_bootstrap_summary(model_confidence_intervals)

    _validate_aligned_predictions(aligned_predictions)

    modality = _validate_modality_importance(
        modality_contribution
    )

    if not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer.")

    labels = tuple(
        str(label) for label in class_labels
    )

    if len(labels) != 4:
        raise ValueError("Exactly four class labels are required.")

    plt = _load_matplotlib()

    try:
        from sklearn.metrics import confusion_matrix

    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for confusion-matrix figure generation."
        ) from exc

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    paths = {
        key: destination / filename
        for key, filename in FIGURE_FILENAMES.items()
    }

    # --------------------------------------------------
    # FIGURE 1: Model performance with 95% bootstrap CI
    # --------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    try:
        for axis, metric, title in (
            (axes[0], "accuracy", "Test Accuracy"),
            (axes[1], "macro_f1", "Test Macro-F1"),
        ):

            data = model_confidence_intervals.loc[
                model_confidence_intervals["Metric"] == metric
            ].copy()

            data["Model"] = pd.Categorical(
                data["Model"],
                categories=list(MODEL_ORDER),
                ordered=True,
            )

            data = data.sort_values("Model")

            estimates = pd.to_numeric(
                data["Test_Estimate"]
            ).to_numpy(dtype=float)

            lower_error = (
                estimates
                - pd.to_numeric(data["CI_95_Lower"]).to_numpy(dtype=float)
            )

            upper_error = (
                pd.to_numeric(data["CI_95_Upper"]).to_numpy(dtype=float)
                - estimates
            )

            x_positions = np.arange(len(data))

            axis.bar(
                x_positions,
                estimates,
            )

            axis.errorbar(
                x_positions,
                estimates,
                yerr=np.vstack([
                    lower_error,
                    upper_error,
                ]),
                fmt="none",
                capsize=5,
            )

            axis.set_xticks(x_positions)

            axis.set_xticklabels(
                data["Model"].astype(str),
                rotation=15,
            )

            axis.set_ylim(0, 1)
            axis.set_ylabel("Score")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)

        figure.suptitle(
            "Model Performance with 95% Subject-Level Bootstrap CI"
        )

        figure.tight_layout()

        figure.savefig(
            paths["performance"],
            dpi=dpi,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)

    # --------------------------------------------------
    # FIGURE 2: Group-level confusion matrices
    # --------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
    )

    image = None

    try:
        for axis, model_name in zip(
            axes,
            MODEL_ORDER,
        ):

            prediction_column = PREDICTION_COLUMNS[model_name]

            matrix = confusion_matrix(
                aligned_predictions["true_grade"],
                aligned_predictions[prediction_column],
                labels=list(GRADE_LABELS),
            )

            image = axis.imshow(
                matrix,
                interpolation="nearest",
            )

            axis.set_title(
                f"{model_name} Confusion Matrix"
            )

            axis.set_xlabel("Predicted Grade")
            axis.set_ylabel("True Grade")

            axis.set_xticks(range(4))
            axis.set_yticks(range(4))

            axis.set_xticklabels(labels)
            axis.set_yticklabels(labels)

            for row in range(4):
                for column in range(4):
                    axis.text(
                        column,
                        row,
                        str(int(matrix[row, column])),
                        ha="center",
                        va="center",
                    )

        figure.colorbar(
            image,
            ax=list(axes),
            shrink=0.8,
        )

        figure.suptitle(
            "Group-Level Test Confusion Matrices"
        )

        figure.tight_layout()

        figure.savefig(
            paths["confusion_matrices"],
            dpi=dpi,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)

    # --------------------------------------------------
    # FIGURE 3: SHAP modality contribution
    # --------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    try:
        modalities = modality["modality"].tolist()

        contributions = modality[
            "contribution_percent"
        ].to_numpy(dtype=float)

        x_positions = np.arange(len(modalities))

        axis.bar(
            x_positions,
            contributions,
        )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(modalities)

        axis.set_ylabel(
            "Mean Absolute SHAP Contribution (%)"
        )

        axis.set_title(
            "Hybrid Model: CNN vs Radiomics Contribution"
        )

        axis.set_ylim(0, 110)
        axis.grid(axis="y", alpha=0.25)

        for index, value in enumerate(contributions):
            axis.text(
                index,
                value + 2,
                f"{value:.1f}%",
                ha="center",
            )

        figure.tight_layout()

        figure.savefig(
            paths["shap_modality"],
            dpi=dpi,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)

    missing_outputs = [
        str(path)
        for path in paths.values()
        if not path.is_file() or path.stat().st_size == 0
    ]

    if missing_outputs:
        raise RuntimeError(
            "Figure generation did not produce all expected files: "
            + ", ".join(missing_outputs)
        )

    return paths
