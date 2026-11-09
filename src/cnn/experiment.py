"""Build and save the reproducible CNN experiment configuration.

This module converts Original Notebook Cell 120 into reusable code.
It combines static experiment settings from ``src.cnn.config.CONFIG`` with
the checkpoint and final test metrics produced by the training pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import CONFIG


def build_cnn_experiment_config(
    checkpoint: Mapping[str, Any],
    test_results: Mapping[str, Any],
) -> dict:
    """Build the experiment JSON payload from actual run outputs.

    Metrics are read from the supplied best checkpoint and test evaluation.
    They are not hard-coded, so the saved configuration always describes
    the run whose checkpoint and test results were passed to this function.
    """
    required_checkpoint = {"epoch", "best_val_group_macro_f1"}
    missing_checkpoint = required_checkpoint.difference(checkpoint)
    if missing_checkpoint:
        raise ValueError(
            "Checkpoint is missing keys: "
            + ", ".join(sorted(missing_checkpoint))
        )

    required_metrics = {
        "group_macro_f1",
        "group_accuracy",
        "group_balanced_accuracy",
        "image_accuracy",
        "image_macro_f1",
    }
    missing_metrics = required_metrics.difference(test_results)
    if missing_metrics:
        raise ValueError(
            "Test results are missing keys: "
            + ", ".join(sorted(missing_metrics))
        )

    try:
        best_epoch = int(checkpoint["epoch"])
        best_validation_f1 = float(
            checkpoint["best_val_group_macro_f1"]
        )
        test_metrics = {
            "test_group_macro_f1": float(test_results["group_macro_f1"]),
            "test_group_accuracy": float(test_results["group_accuracy"]),
            "test_group_balanced_accuracy": float(
                test_results["group_balanced_accuracy"]
            ),
            "test_image_accuracy": float(test_results["image_accuracy"]),
            "test_image_macro_f1": float(test_results["image_macro_f1"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint and test metric values must be numeric."
        ) from exc

    all_scores = [best_validation_f1, *test_metrics.values()]
    if best_epoch < 1:
        raise ValueError("Checkpoint epoch must be at least 1.")
    if not all(math.isfinite(value) for value in all_scores):
        raise ValueError("All saved metrics must be finite numbers.")

    return {
        "experiment_name": "Dataset_B_CNN_Baseline",
        "dataset": "FSHD Ultrasound",
        "classification_task": "Heckmatt Grade Classification",
        "architecture": "ResNet18",
        "pretrained_weights": "ImageNet",
        "input_channels": 1,
        "image_size": int(CONFIG.image_size),
        "num_classes": int(CONFIG.num_classes),
        "class_names": ["G1", "G2", "G3", "G4"],
        "batch_size": int(CONFIG.batch_size),
        "optimizer": "AdamW",
        "learning_rate": float(CONFIG.learning_rate),
        "weight_decay": float(CONFIG.weight_decay),
        "loss_function": "Weighted CrossEntropyLoss",
        "maximum_epochs": int(CONFIG.max_epochs),
        "early_stopping_patience": int(CONFIG.early_stopping_patience),
        "random_seed": int(CONFIG.random_seed),
        "best_epoch": best_epoch,
        "best_validation_group_macro_f1": best_validation_f1,
        **test_metrics,
        "primary_evaluation": "Group-level mean probability aggregation",
        "secondary_evaluation": "Image-level classification",
        "split_strategy": "Subject-level split",
        "split_seed": int(CONFIG.random_seed),
    }


def save_cnn_experiment_config(
    checkpoint: Mapping[str, Any],
    test_results: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    """Save ``cnn_experiment_config.json`` to the supplied CNN output folder."""
    output_path = Path(output_dir).expanduser()

    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(
            f"CNN output path is not a directory: {output_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    config = build_cnn_experiment_config(checkpoint, test_results)

    config_path = output_path / "cnn_experiment_config.json"
    config_path.write_text(
        json.dumps(config, indent=4),
        encoding="utf-8",
    )

    # Read the saved artifact back to detect incomplete/corrupt writes.
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if saved_config != config:
        raise RuntimeError(
            f"Saved experiment configuration failed verification: {config_path}"
        )

    return config_path
