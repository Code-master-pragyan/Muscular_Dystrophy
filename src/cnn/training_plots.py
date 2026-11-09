"""Generate training-history plots for the Dataset B CNN baseline.

This module converts Original Notebook Cell 121 into reusable code.
It preserves the three plots and filenames used by the notebook:
training loss, training accuracy, and validation group-level macro-F1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


PLOT_SPECS = (
    (
        "train_loss",
        "CNN Training Loss Across Epochs",
        "Training Loss",
        "training_loss.png",
    ),
    (
        "train_accuracy",
        "CNN Training Accuracy Across Epochs",
        "Training Accuracy",
        "training_accuracy.png",
    ),
    (
        "val_group_macro_f1",
        "Validation Group-Level Macro-F1 Across Epochs",
        "Group Macro-F1",
        "validation_group_macro_f1.png",
    ),
)

REQUIRED_COLUMNS = {"epoch", *(spec[0] for spec in PLOT_SPECS)}


def generate_training_plots(
    history: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Path]:
    """Generate and save the notebook's three training-history graphs.

    Parameters
    ----------
    history:
        Training history DataFrame returned by ``train_cnn`` or path to
        the saved ``training_history.csv``.
    output_dir:
        Directory in which the PNG graphs will be written. The directory
        is created if it does not exist.
    dpi:
        Export resolution. Defaults to 300, as in the source notebook.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from history metric column names to their generated PNG paths.
    """
    if isinstance(history, pd.DataFrame):
        history_df = history.copy()
    else:
        history_path = Path(history).expanduser()
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Training history CSV not found: {history_path}"
            )
        history_df = pd.read_csv(history_path)

    missing = REQUIRED_COLUMNS.difference(history_df.columns)
    if missing:
        raise ValueError(
            "Training history is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if history_df.empty:
        raise ValueError("Training history is empty.")

    numeric_columns = sorted(REQUIRED_COLUMNS)
    for column in numeric_columns:
        history_df[column] = pd.to_numeric(
            history_df[column],
            errors="coerce",
        )
        if history_df[column].isna().any():
            raise ValueError(
                f"Training history column {column!r} contains "
                "missing or non-numeric values."
            )

    if (history_df["epoch"] < 1).any():
        raise ValueError("Epoch values must be positive.")
    if history_df["epoch"].duplicated().any():
        raise ValueError("Training history contains duplicate epoch values.")

    history_df = history_df.sort_values("epoch").reset_index(drop=True)

    if not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer.")

    # Import lazily so the training pipeline does not require plotting
    # libraries unless these figures are requested.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate training plots. "
            "Install it with: pip install matplotlib"
        ) from exc

    graph_dir = Path(output_dir).expanduser()
    graph_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    for column, title, ylabel, filename in PLOT_SPECS:
        figure, axis = plt.subplots(figsize=(9, 5))
        try:
            axis.plot(
                history_df["epoch"],
                history_df[column],
                marker="o",
            )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_xticks(history_df["epoch"])
            axis.grid(True, alpha=0.3)
            figure.tight_layout()

            output_path = graph_dir / filename
            figure.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
            )
            generated[column] = output_path
        finally:
            plt.close(figure)

    return generated
