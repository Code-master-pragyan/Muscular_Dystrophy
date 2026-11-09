"""SHAP explainability and feature-usage audit for the hybrid XGBoost model.

Converts the original Colab SHAP blocks into a reusable module:
- calculate multi-class TreeSHAP values for the selected hybrid booster;
- compute global and class-wise mean absolute SHAP importance;
- calculate CNN/radiomics modality contribution;
- map XGBoost split keys such as ``f50`` back to the original feature
  column names before classifying modality;
- save CSV, compressed NPZ, and JSON audit artifacts.

Important: XGBoost may expose split keys as f0, f1, ... when the model was
trained from NumPy arrays. These are positional indices into the feature
matrix, not modality labels. This module maps them to the supplied
feature_names to avoid misclassifying all fN keys as radiomics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_CLASS_NAMES = ("G1", "G2", "G3", "G4")
CNN_PREFIX = "cnn_feature_"
RADIOMICS_PREFIX = "original_"


def _resolve_booster(model):
    """Accept either an XGBClassifier or a raw XGBoost Booster."""
    if hasattr(model, "get_booster"):
        return model.get_booster()
    if hasattr(model, "get_score") and hasattr(model, "num_boosted_rounds"):
        return model
    raise TypeError(
        "model must be an XGBClassifier or xgboost.Booster."
    )


def _normalise_shap_array(
    shap_values,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Convert supported SHAP multi-class outputs to (N, F, C)."""
    if isinstance(shap_values, list):
        if len(shap_values) != n_classes:
            raise ValueError(
                f"Expected {n_classes} SHAP class arrays, got {len(shap_values)}."
            )
        arrays = [np.asarray(values, dtype=np.float64) for values in shap_values]
        if any(array.shape != (n_samples, n_features) for array in arrays):
            raise ValueError(
                "Each class SHAP array must have shape "
                f"{(n_samples, n_features)}."
            )
        result = np.stack(arrays, axis=-1)
    else:
        result = np.asarray(shap_values, dtype=np.float64)

        if result.shape == (n_samples, n_features, n_classes):
            pass
        elif result.shape == (n_classes, n_samples, n_features):
            result = np.transpose(result, (1, 2, 0))
        else:
            raise ValueError(
                "Unsupported SHAP array shape. Expected "
                f"{(n_samples, n_features, n_classes)} or "
                f"{(n_classes, n_samples, n_features)}, got {result.shape}."
            )

    if not np.isfinite(result).all():
        raise ValueError("SHAP values contain NaN or infinite values.")
    return result


def _map_booster_feature_name(
    booster_feature: str,
    feature_names: Sequence[str],
) -> str:
    """Map a booster feature key (fN or named) to the original feature name."""
    key = str(booster_feature)

    if key in feature_names:
        return key

    match = re.fullmatch(r"f(\d+)", key)
    if match:
        index = int(match.group(1))
        if index >= len(feature_names):
            raise ValueError(
                f"Booster split key {key!r} points outside the feature list "
                f"of length {len(feature_names)}."
            )
        return str(feature_names[index])

    raise ValueError(
        f"Cannot map booster feature key {key!r} to the supplied feature names."
    )


def build_shap_importance_tables(
    shap_array: np.ndarray,
    feature_names: Sequence[str],
    *,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build global/class-wise SHAP importance and modality contribution."""
    values = np.asarray(shap_array, dtype=np.float64)
    names = [str(name) for name in feature_names]
    classes = [str(name) for name in class_names]

    if values.ndim != 3:
        raise ValueError("shap_array must have shape (samples, features, classes).")
    if values.shape[1] != len(names):
        raise ValueError(
            f"SHAP has {values.shape[1]} features but {len(names)} names supplied."
        )
    if values.shape[2] != len(classes):
        raise ValueError(
            f"SHAP has {values.shape[2]} classes but {len(classes)} names supplied."
        )
    if len(names) != len(set(names)):
        raise ValueError("feature_names must be unique.")
    if not np.isfinite(values).all():
        raise ValueError("SHAP array contains non-finite values.")

    class_importance = np.mean(np.abs(values), axis=0)
    global_importance = np.mean(class_importance, axis=1)

    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": global_importance,
    })
    for class_index, class_name in enumerate(classes):
        importance[f"{class_name}_importance"] = class_importance[:, class_index]

    def get_modality(feature_name: str) -> str:
        if feature_name.startswith(CNN_PREFIX):
            return "CNN"
        if feature_name.startswith(RADIOMICS_PREFIX):
            return "Radiomics"
        raise ValueError(
            f"Unrecognized hybrid feature prefix for {feature_name!r}."
        )

    importance["modality"] = importance["feature"].map(get_modality)
    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    modality = (
        importance.groupby("modality", as_index=False)["mean_abs_shap"]
        .sum()
        .set_index("modality")
        .reindex(["CNN", "Radiomics"], fill_value=0.0)
        .reset_index()
    )
    total_importance = float(modality["mean_abs_shap"].sum())
    if total_importance > 0:
        modality["contribution_percent"] = (
            modality["mean_abs_shap"] / total_importance * 100.0
        )
    else:
        modality["contribution_percent"] = 0.0

    activity = {}
    for modality_name in ("CNN", "Radiomics"):
        indices = [
            index for index, name in enumerate(names)
            if get_modality(name) == modality_name
        ]
        subset = np.abs(values[:, indices, :]) if indices else np.empty((values.shape[0], 0, values.shape[2]))
        activity[modality_name] = {
            "feature_count": int(len(indices)),
            "nonzero_shap_values": int(np.count_nonzero(subset)),
            "total_absolute_shap": float(subset.sum()),
            "contribution_percent": float(
                modality.loc[
                    modality["modality"] == modality_name,
                    "contribution_percent",
                ].iloc[0]
            ),
        }

    return importance, modality, activity


def build_tree_feature_usage_table(
    booster,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, dict]:
    """Map actual XGBoost split usage back to named CNN/radiomics features."""
    split_counts = booster.get_score(importance_type="weight")
    mapped_rows = []

    for booster_key, split_count in split_counts.items():
        original_name = _map_booster_feature_name(
            str(booster_key),
            feature_names,
        )
        if original_name.startswith(CNN_PREFIX):
            modality = "CNN"
        elif original_name.startswith(RADIOMICS_PREFIX):
            modality = "Radiomics"
        else:
            raise ValueError(
                f"Unrecognized modality for mapped feature {original_name!r}."
            )
        mapped_rows.append({
            "booster_feature_key": str(booster_key),
            "feature": original_name,
            "modality": modality,
            "split_count": int(split_count),
        })

    usage = pd.DataFrame(
        mapped_rows,
        columns=[
            "booster_feature_key",
            "feature",
            "modality",
            "split_count",
        ],
    )
    if not usage.empty:
        usage = usage.sort_values(
            ["split_count", "feature"],
            ascending=[False, True],
        ).reset_index(drop=True)

    summary = {
        "distinct_features_used": int(len(usage)),
        "total_tree_splits": int(usage["split_count"].sum()) if not usage.empty else 0,
        "cnn_features_used": int((usage["modality"] == "CNN").sum()) if not usage.empty else 0,
        "radiomics_features_used": int((usage["modality"] == "Radiomics").sum()) if not usage.empty else 0,
        "cnn_split_count": int(
            usage.loc[usage["modality"] == "CNN", "split_count"].sum()
        ) if not usage.empty else 0,
        "radiomics_split_count": int(
            usage.loc[usage["modality"] == "Radiomics", "split_count"].sum()
        ) if not usage.empty else 0,
    }
    return usage, summary


def calculate_hybrid_shap_explainability(
    selected_model,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    *,
    selected_rounds: int = 13,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> dict:
    """Calculate selected-booster TreeSHAP values and feature usage audits.

    For the original notebook hybrid model, use the saved 13-round selected
    booster and ``selected_rounds=13``. If XGBoost split keys are positional
    fN tokens, they are mapped through ``feature_names`` before modality
    counts are calculated.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explainability. Install it with: pip install shap"
        ) from exc

    booster = _resolve_booster(selected_model)
    available_rounds = int(booster.num_boosted_rounds())
    if selected_rounds < 1 or selected_rounds > available_rounds:
        raise ValueError(
            f"selected_rounds must be between 1 and {available_rounds}; "
            f"received {selected_rounds}."
        )

    if isinstance(X_test, pd.DataFrame):
        test_frame = X_test.copy()
        names = [str(column) for column in test_frame.columns]
        if feature_names is not None and list(map(str, feature_names)) != names:
            raise ValueError(
                "feature_names does not match the columns of X_test."
            )
        values = test_frame.apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            raise ValueError("X_test contains missing/non-numeric values.")
        matrix = values.to_numpy(dtype=np.float32)
    else:
        matrix = np.asarray(X_test, dtype=np.float32)
        if feature_names is None:
            raise ValueError(
                "feature_names is required when X_test is a NumPy array."
            )
        names = [str(name) for name in feature_names]
        test_frame = pd.DataFrame(matrix, columns=names)

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("X_test must be a non-empty 2D feature matrix.")
    if matrix.shape[1] != len(names):
        raise ValueError("X_test column count does not match feature_names.")
    if len(names) != len(set(names)):
        raise ValueError("feature_names must be unique.")
    if not np.isfinite(matrix).all():
        raise ValueError("X_test contains NaN or infinite values.")

    class_names = tuple(str(name) for name in class_names)
    if len(class_names) != 4:
        raise ValueError("Exactly four class names are required.")

    if hasattr(booster, "num_features") and booster.num_features() != matrix.shape[1]:
        raise ValueError(
            f"Model expects {booster.num_features()} features, "
            f"but X_test has {matrix.shape[1]}."
        )

    # Select the first N rounds of the saved model to match the notebook.
    selected_booster = booster[:selected_rounds]
    explainer = shap.TreeExplainer(selected_booster)

    started = perf_counter()
    raw_shap_values = explainer.shap_values(test_frame)
    elapsed = perf_counter() - started

    shap_array = _normalise_shap_array(
        raw_shap_values,
        n_samples=matrix.shape[0],
        n_features=matrix.shape[1],
        n_classes=len(class_names),
    )

    importance, modality, shap_activity = build_shap_importance_tables(
        shap_array,
        names,
        class_names=class_names,
    )
    usage, usage_summary = build_tree_feature_usage_table(
        selected_booster,
        names,
    )

    interpretation = (
        "The selected booster used CNN features only; no radiomics features "
        "were used in tree splits."
        if usage_summary["cnn_features_used"] > 0
        and usage_summary["radiomics_features_used"] == 0
        else
        "The selected booster used radiomics features only; no CNN features "
        "were used in tree splits."
        if usage_summary["radiomics_features_used"] > 0
        and usage_summary["cnn_features_used"] == 0
        else
        "The selected booster used features from both modalities."
        if usage_summary["cnn_features_used"] > 0
        and usage_summary["radiomics_features_used"] > 0
        else
        "No tree split features were reported by the selected booster."
    )

    metadata = {
        "model": "XGBoost CNN-Radiomics Hybrid",
        "explainer": "SHAP TreeExplainer",
        "shap_version": getattr(shap, "__version__", "unknown"),
        "selected_boosting_rounds": int(selected_rounds),
        "available_boosting_rounds_in_source_model": available_rounds,
        "test_groups": int(matrix.shape[0]),
        "total_features": int(matrix.shape[1]),
        "classes": list(class_names),
        "cnn_features": int(shap_activity["CNN"]["feature_count"]),
        "radiomics_features": int(shap_activity["Radiomics"]["feature_count"]),
        "cnn_features_used_in_trees": usage_summary["cnn_features_used"],
        "radiomics_features_used_in_trees": usage_summary["radiomics_features_used"],
        "total_tree_splits": usage_summary["total_tree_splits"],
        "cnn_shap_contribution_percent": shap_activity["CNN"]["contribution_percent"],
        "radiomics_shap_contribution_percent": shap_activity["Radiomics"]["contribution_percent"],
        "cnn_nonzero_shap_values": shap_activity["CNN"]["nonzero_shap_values"],
        "radiomics_nonzero_shap_values": shap_activity["Radiomics"]["nonzero_shap_values"],
        "shap_output": "Raw model output (TreeExplainer default)",
        "calculation_time_seconds": float(elapsed),
        "interpretation": interpretation,
    }

    return {
        "shap_values": shap_array,
        "feature_names": names,
        "global_classwise_importance": importance,
        "modality_contribution": modality,
        "feature_usage": usage,
        "feature_usage_summary": usage_summary,
        "shap_activity": shap_activity,
        "metadata": metadata,
        "selected_booster": selected_booster,
    }


def save_hybrid_shap_artifacts(
    explainability: dict,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save the five SHAP/explainability artifacts used by the notebook."""
    required = {
        "shap_values",
        "feature_names",
        "global_classwise_importance",
        "modality_contribution",
        "feature_usage",
        "metadata",
    }
    missing = required.difference(explainability)
    if missing:
        raise ValueError(
            "Explainability result is missing keys: "
            + ", ".join(sorted(missing))
        )

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    paths = {
        "global_classwise_importance": destination
        / "Hybrid_SHAP_Global_Classwise_Importance.csv",
        "feature_usage": destination
        / "Hybrid_XGBoost_Corrected_Feature_Usage.csv",
        "modality_contribution": destination
        / "Hybrid_SHAP_Modality_Contribution.csv",
        "raw_shap_values": destination / "Hybrid_Test_SHAP_Values.npz",
        "metadata": destination / "Hybrid_SHAP_Audit_Metadata.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing SHAP artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Pass overwrite=True to replace them."
        )

    explainability["global_classwise_importance"].to_csv(
        paths["global_classwise_importance"],
        index=False,
    )
    explainability["feature_usage"].to_csv(
        paths["feature_usage"],
        index=False,
    )
    explainability["modality_contribution"].to_csv(
        paths["modality_contribution"],
        index=False,
    )
    np.savez_compressed(
        paths["raw_shap_values"],
        shap_values=np.asarray(explainability["shap_values"]),
        feature_names=np.asarray(explainability["feature_names"]),
    )
    paths["metadata"].write_text(
        json.dumps(explainability["metadata"], indent=4),
        encoding="utf-8",
    )

    return paths
