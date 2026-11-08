"""PyTorch dataset and image transforms for Dataset B ultrasound images."""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CONFIG


def build_image_transform():
    """Create the grayscale image transform used in the original CNN run."""
    return transforms.Compose(
        [
            transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[CONFIG.normalize_mean],
                std=[CONFIG.normalize_std],
            ),
        ]
    )


class FSHDUltrasoundDataset(Dataset):
    """Load labelled ultrasound images from a split DataFrame.

    Required DataFrame columns:
        image_path, label, subject_id, group_id, filename

    Labels must already be zero-based class indices (0–3), as in the
    original CNN training pipeline.
    """

    REQUIRED_COLUMNS = {
        "image_path",
        "label",
        "subject_id",
        "group_id",
        "filename",
    }

    def __init__(self, dataframe, transform: Optional[object] = None):
        missing = self.REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing:
            raise ValueError(
                "DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform if transform is not None else build_image_transform()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        image_path = Path(row["image_path"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Ultrasound image not found: {image_path}")

        # The original experiment loads each ultrasound image as grayscale.
        with Image.open(image_path) as source:
            image = source.convert("L")

        image = self.transform(image)

        label = int(row["label"])
        if not 0 <= label < CONFIG.num_classes:
            raise ValueError(
                f"Expected zero-based label in [0, {CONFIG.num_classes - 1}], "
                f"received {label} at DataFrame row {index}."
            )

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": int(row["subject_id"]),
            "group_id": str(row["group_id"]),
            "filename": str(row["filename"]),
        }
