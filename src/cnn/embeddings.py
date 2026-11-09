"""Extract frozen ResNet18 embeddings for Dataset B CNN/radiomics fusion.

This module consolidates Original Notebook Cells 157–169:
- restore the trained grayscale ResNet18 checkpoint;
- replace the classification head with Identity to expose 512-D embeddings;
- extract embeddings independently for train, validation, and test;
- save metadata plus CNN features in CSV chunks every 20 batches.

The approved subject-level split is supplied by the caller. This module
never creates a new split and does not use ultrasound masks during CNN
embedding extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import CONFIG
from .dataset import build_image_transform
from .model import build_resnet18
from .checkpoint import load_best_checkpoint


PARTITIONS = ("train", "validation", "test")
METADATA_COLUMNS = (
    "filename",
    "subject_id",
    "muscle_code",
    "side_code",
    "group_id",
    "heckmatt_grade",
    "partition",
)
REQUIRED_COLUMNS = {
    "image_path",
    *METADATA_COLUMNS,
}
EMBEDDING_DIMENSION = 512


class CNNEmbeddingDataset(Dataset):
    """Load grayscale images and retain the metadata needed for fusion."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        missing = REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing:
            raise ValueError(
                "Embedding DataFrame is missing columns: "
                + ", ".join(sorted(missing))
            )
        if dataframe.empty:
            raise ValueError("Cannot extract embeddings from an empty DataFrame.")
        if dataframe["filename"].astype(str).duplicated().any():
            raise ValueError("Embedding DataFrame contains duplicate filenames.")
        if dataframe[list(METADATA_COLUMNS)].isna().any().any():
            raise ValueError("Embedding metadata contains missing values.")

        grades = pd.to_numeric(dataframe["heckmatt_grade"], errors="coerce")
        if grades.isna().any() or not grades.isin([1, 2, 3, 4]).all():
            raise ValueError("Heckmatt grades must be integers from 1 through 4.")

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.dataframe["heckmatt_grade"] = grades.astype(int)
        self.transform = transform if transform is not None else build_image_transform()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        image_path = Path(str(row["image_path"]))

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Ultrasound image not found: {image_path}"
            )

        with Image.open(image_path) as source:
            image = source.convert("L")

        image = self.transform(image)

        return {
            "image": image,
            "filename": str(row["filename"]),
            "subject_id": row["subject_id"],
            "muscle_code": row["muscle_code"],
            "side_code": row["side_code"],
            "group_id": str(row["group_id"]),
            "heckmatt_grade": int(row["heckmatt_grade"]),
            "partition": str(row["partition"]),
        }


def load_resnet18_feature_extractor(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Restore the trained ResNet18 and expose its 512-dimensional features."""
    classifier = build_resnet18(
        num_classes=CONFIG.num_classes,
        pretrained=False,
    )
    classifier, checkpoint = load_best_checkpoint(
        classifier,
        checkpoint_path,
        device,
    )

    classifier.fc = nn.Identity()

    for parameter in classifier.parameters():
        parameter.requires_grad = False

    classifier = classifier.to(device)
    classifier.eval()
    return classifier, checkpoint


def extract_partition_embeddings(
    model: nn.Module,
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    partition: str,
    device: torch.device,
    *,
    batch_size: int = 16,
    save_every_batches: int = 20,
    num_workers: int = 0,
    transform=None,
    overwrite: bool = False,
) -> list[Path]:
    """Extract and save one partition's 512-D embeddings as sequential CSV chunks.

    Chunk names follow the notebook, for example ``train_batch_0020.csv``.
    Each output chunk contains the source metadata followed by
    ``cnn_feature_1`` through ``cnn_feature_512``.
    """
    partition = str(partition).strip().lower()
    if partition not in PARTITIONS:
        raise ValueError(
            f"partition must be one of {PARTITIONS}; received {partition!r}."
        )
    if batch_size < 1 or save_every_batches < 1:
        raise ValueError("batch_size and save_every_batches must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if "partition" not in dataframe.columns:
        raise ValueError("DataFrame must contain a 'partition' column.")
    actual_partitions = set(dataframe["partition"].astype(str).str.lower())
    if actual_partitions != {partition}:
        raise ValueError(
            f"Expected only partition {partition!r}; found {sorted(actual_partitions)}."
        )

    dataset = CNNEmbeddingDataset(dataframe, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    destination = Path(output_dir).expanduser() / partition
    destination.mkdir(parents=True, exist_ok=True)

    existing_chunks = list(destination.glob(f"{partition}_batch_*.csv"))
    if existing_chunks and not overwrite:
        raise FileExistsError(
            f"{len(existing_chunks)} {partition} chunk file(s) already exist "
            f"in {destination}. Set overwrite=True to replace matching chunks."
        )
    if overwrite:
        for old_chunk in existing_chunks:
            old_chunk.unlink()

    model = model.to(device)
    model.eval()

    feature_buffer = []
    metadata_buffer = []
    written_paths: list[Path] = []
    feature_columns = [
        f"cnn_feature_{index + 1}"
        for index in range(EMBEDDING_DIMENSION)
    ]

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            features = model(images)

            if not torch.is_tensor(features) or features.ndim != 2:
                raise ValueError(
                    "Feature extractor must return a 2D tensor shaped (B, 512)."
                )
            if features.shape[0] != images.shape[0]:
                raise ValueError("Feature batch size does not match image batch size.")
            if features.shape[1] != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings; "
                    f"received {features.shape[1]}."
                )
            if not torch.isfinite(features).all():
                raise ValueError(
                    f"Non-finite CNN embedding encountered in batch {batch_index}."
                )

            feature_buffer.append(
                features.detach().cpu().numpy().astype(np.float32, copy=False)
            )

            batch_size_actual = int(images.shape[0])
            for index in range(batch_size_actual):
                metadata_buffer.append(
                    {
                        "filename": batch["filename"][index],
                        "subject_id": batch["subject_id"][index].item()
                        if torch.is_tensor(batch["subject_id"])
                        else batch["subject_id"][index],
                        "muscle_code": batch["muscle_code"][index].item()
                        if torch.is_tensor(batch["muscle_code"])
                        else batch["muscle_code"][index],
                        "side_code": batch["side_code"][index].item()
                        if torch.is_tensor(batch["side_code"])
                        else batch["side_code"][index],
                        "group_id": batch["group_id"][index],
                        "heckmatt_grade": int(
                            batch["heckmatt_grade"][index].item()
                            if torch.is_tensor(batch["heckmatt_grade"])
                            else batch["heckmatt_grade"][index]
                        ),
                        "partition": batch["partition"][index],
                    }
                )

            should_save = (
                batch_index % save_every_batches == 0
                or batch_index == len(loader)
            )

            if should_save:
                feature_array = np.concatenate(feature_buffer, axis=0)
                metadata_df = pd.DataFrame(metadata_buffer)

                if len(metadata_df) != len(feature_array):
                    raise RuntimeError(
                        "Metadata and embedding row counts do not match."
                    )

                feature_df = pd.DataFrame(
                    feature_array,
                    columns=feature_columns,
                )
                chunk_df = pd.concat(
                    [
                        metadata_df.reset_index(drop=True),
                        feature_df.reset_index(drop=True),
                    ],
                    axis=1,
                )

                chunk_path = destination / (
                    f"{partition}_batch_{batch_index:04d}.csv"
                )
                chunk_df.to_csv(chunk_path, index=False)
                written_paths.append(chunk_path)

                feature_buffer.clear()
                metadata_buffer.clear()

    return written_paths


def extract_cnn_embedding_partitions(
    model: nn.Module,
    partitions: Mapping[str, pd.DataFrame],
    output_root: str | Path,
    device: torch.device,
    *,
    batch_size: int = 16,
    save_every_batches: int = 20,
    num_workers: int = 0,
    transform=None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Extract embeddings for each supplied approved partition DataFrame."""
    missing = set(PARTITIONS).difference(partitions)
    if missing:
        raise ValueError(
            "Partitions mapping must contain train, validation, and test; "
            f"missing: {sorted(missing)}."
        )

    outputs = {}
    for partition in PARTITIONS:
        outputs[partition] = extract_partition_embeddings(
            model=model,
            dataframe=partitions[partition],
            output_dir=output_root,
            partition=partition,
            device=device,
            batch_size=batch_size,
            save_every_batches=save_every_batches,
            num_workers=num_workers,
            transform=transform,
            overwrite=overwrite,
        )
    return outputs


def load_cnn_embedding_chunks(
    chunk_dir: str | Path,
    partition: str,
) -> pd.DataFrame:
    """Read and concatenate the saved CSV chunks for one partition."""
    partition = str(partition).strip().lower()
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition!r}.")

    directory = Path(chunk_dir).expanduser() / partition
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Embedding chunk directory not found: {directory}"
        )

    chunk_files = sorted(directory.glob(f"{partition}_batch_*.csv"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No {partition} embedding chunks found in {directory}."
        )

    frames = [pd.read_csv(path) for path in chunk_files]
    combined = pd.concat(frames, ignore_index=True)

    required_features = [
        f"cnn_feature_{index + 1}"
        for index in range(EMBEDDING_DIMENSION)
    ]
    missing = set(METADATA_COLUMNS).union(required_features).difference(
        combined.columns
    )
    if missing:
        raise ValueError(
            "Embedding chunks are missing columns: "
            + ", ".join(sorted(missing))
        )

    if combined["filename"].astype(str).duplicated().any():
        raise ValueError(
            f"Duplicate filenames found across {partition} chunks."
        )

    return combined
