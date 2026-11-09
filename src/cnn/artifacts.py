"""Filesystem locations for Dataset B CNN training artifacts.

This module converts the directory setup from Original Notebook Cell 108
into a reusable helper. It does not mount Google Drive or assume a specific
Colab path; callers pass the project root explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CNNArtifactPaths:
    """Resolved paths used by the CNN training run."""

    output_dir: Path
    checkpoint_dir: Path
    history_dir: Path
    best_model_path: Path


def create_cnn_artifact_paths(
    project_root: str | Path,
    *,
    create_directories: bool = True,
) -> CNNArtifactPaths:
    """Resolve CNN output paths and optionally create their directories.

    Parameters
    ----------
    project_root:
        Root ``Muscular_Dystrophy_Project`` directory. The CNN experiment
        folder is created beneath
        ``05_Model_Development/Dataset_B_CNN_Baseline``.
    create_directories:
        When True, create the experiment, checkpoint, and history folders.
        Set False to resolve paths without modifying the filesystem.

    Returns
    -------
    CNNArtifactPaths
        Paths to the CNN output directory, checkpoint directory,
        training-history directory, and best-model checkpoint file.
    """
    root = Path(project_root).expanduser()

    if not root.exists() and not create_directories:
        # A non-existing root is valid when only resolving paths.
        pass
    elif root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")

    output_dir = (
        root
        / "05_Model_Development"
        / "Dataset_B_CNN_Baseline"
    )
    checkpoint_dir = output_dir / "checkpoints"
    history_dir = output_dir / "training_history"
    best_model_path = checkpoint_dir / "resnet18_best_model.pth"

    if create_directories:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)

    return CNNArtifactPaths(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        history_dir=history_dir,
        best_model_path=best_model_path,
    )
