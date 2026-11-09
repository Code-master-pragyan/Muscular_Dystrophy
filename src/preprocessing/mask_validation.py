"""Validation helpers for processed FSHD ultrasound masks.

This module converts the processed-mask loading check from the Colab workflow
into a reusable function. By default, it checks the first two labelled records
from each partition (train, validation, and test), matching the deterministic
spot-check performed in the notebook.

The ultrasound images are loaded as grayscale and scaled to float32 [0, 1].
Processed masks are loaded without rescaling so their expected binary values
{0, 1} are preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {
    "filename",
    "image_path",
    "partition",
    "heckmatt_grade",
}

DEFAULT_PARTITIONS = ("train", "validation", "test")
DEFAULT_IMAGE_SHAPE = (512, 512)
EXPECTED_MASK_VALUES = {0, 1}


def validate_processed_mask_samples(
    labelled_df: pd.DataFrame,
    processed_masks_dir: Union[str, Path],
    *,
    samples_per_partition: Optional[int] = 2,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
    expected_shape: Optional[tuple[int, int]] = DEFAULT_IMAGE_SHAPE,
) -> pd.DataFrame:
    """Check image/processed-mask pairs for selected records in each partition.

    Args:
        labelled_df: Labelled manifest joined to the approved subject split.
            Must contain ``filename``, ``image_path``, ``partition``, and
            ``heckmatt_grade`` columns.
        processed_masks_dir: Directory containing processed masks named exactly
            like their corresponding ultrasound images.
        samples_per_partition: Number of leading records to check in each
            partition. The default value (2) matches the Colab spot check.
            Pass ``None`` to check every record in each partition.
        partitions: Partitions to inspect, in the requested order.
        expected_shape: Expected 2D image and mask shape. Defaults to 512x512.
            Pass ``None`` to skip the fixed-shape check while retaining the
            2D-image and image/mask shape checks.

    Returns:
        A DataFrame with one validation result per attempted image/mask pair.
        A failed file load is recorded in ``error`` and marked as not passed.
        The function does not modify source images, masks, or metadata.

    Raises:
        TypeError: If ``labelled_df`` is not a pandas DataFrame.
        ValueError: If required columns, partition values, or arguments are invalid.
        NotADirectoryError: If the processed-mask directory does not exist.
    """
    if not isinstance(labelled_df, pd.DataFrame):
        raise TypeError("labelled_df must be a pandas DataFrame.")

    missing_columns = REQUIRED_COLUMNS.difference(labelled_df.columns)
    if missing_columns:
        raise ValueError(
            "labelled_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if samples_per_partition is not None and samples_per_partition < 1:
        raise ValueError("samples_per_partition must be >= 1 or None.")

    if not partitions or len(set(partitions)) != len(partitions):
        raise ValueError("partitions must be a non-empty sequence without duplicates.")

    mask_dir = Path(processed_masks_dir)
    if not mask_dir.is_dir():
        raise NotADirectoryError(
            f"Processed masks directory not found: {mask_dir}"
        )

    observed_partitions = set(labelled_df["partition"].dropna().astype(str))
    missing_partitions = set(partitions).difference(observed_partitions)
    if missing_partitions:
        raise ValueError(
            "labelled_df has no records for partition(s): "
            + ", ".join(sorted(missing_partitions))
        )

    results: list[dict] = []

    for partition_name in partitions:
        partition_df = labelled_df[
            labelled_df["partition"].astype(str) == partition_name
        ]
        if samples_per_partition is not None:
            partition_df = partition_df.head(samples_per_partition)

        for row_index, row in partition_df.iterrows():
            filename = str(row["filename"])
            image_path = Path(str(row["image_path"]))
            mask_path = mask_dir / filename

            result = {
                "row_index": int(row_index),
                "filename": filename,
                "partition": partition_name,
                "heckmatt_grade": row["heckmatt_grade"],
                "image_path": str(image_path),
                "processed_mask_path": str(mask_path),
                "image_shape": None,
                "image_dtype": None,
                "image_min": np.nan,
                "image_max": np.nan,
                "mask_shape": None,
                "mask_dtype": None,
                "mask_values": None,
                "mask_foreground_pixels": np.nan,
                "mask_coverage_percent": np.nan,
                "image_ok": False,
                "mask_ok": False,
                "pair_ok": False,
                "mask_nonempty": False,
                "error": None,
            }

            try:
                if not image_path.is_file():
                    raise FileNotFoundError(f"Ultrasound image not found: {image_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Processed mask not found: {mask_path}")

                # Match the notebook: read image as grayscale and scale to [0, 1].
                with Image.open(image_path) as source_image:
                    image_array = np.asarray(source_image.convert("L"))
                image_float = image_array.astype(np.float32) / 255.0

                # Do not convert or rescale the processed mask; preserve labels.
                with Image.open(mask_path) as source_mask:
                    mask_array = np.asarray(source_mask)

                mask_values = set(np.unique(mask_array).tolist())
                image_shape_ok = (
                    image_array.ndim == 2
                    and (expected_shape is None or image_array.shape == expected_shape)
                )
                image_range_ok = (
                    image_float.dtype == np.float32
                    and np.isfinite(image_float).all()
                    and image_float.min() >= 0.0
                    and image_float.max() <= 1.0
                )
                image_ok = image_shape_ok and image_range_ok

                mask_shape_ok = (
                    mask_array.ndim == 2
                    and (expected_shape is None or mask_array.shape == expected_shape)
                )
                mask_labels_ok = (
                    len(mask_values) > 0
                    and mask_values.issubset(EXPECTED_MASK_VALUES)
                )
                mask_ok = mask_shape_ok and mask_labels_ok

                pair_ok = (
                    image_ok
                    and mask_ok
                    and image_array.shape == mask_array.shape
                )
                foreground_pixels = int(np.count_nonzero(mask_array == 1))
                mask_nonempty = foreground_pixels > 0

                result.update(
                    {
                        "image_shape": tuple(int(v) for v in image_array.shape),
                        "image_dtype": str(image_float.dtype),
                        "image_min": float(image_float.min()),
                        "image_max": float(image_float.max()),
                        "mask_shape": tuple(int(v) for v in mask_array.shape),
                        "mask_dtype": str(mask_array.dtype),
                        "mask_values": sorted(mask_values),
                        "mask_foreground_pixels": foreground_pixels,
                        "mask_coverage_percent": round(
                            100.0 * foreground_pixels / mask_array.size, 4
                        ),
                        "image_ok": bool(image_ok),
                        "mask_ok": bool(mask_ok),
                        "pair_ok": bool(pair_ok),
                        "mask_nonempty": bool(mask_nonempty),
                    }
                )

            except Exception as exc:  # Preserve failures so the full sample report is returned.
                result["error"] = f"{type(exc).__name__}: {exc}"

            results.append(result)

    report = pd.DataFrame(results)
    if not report.empty:
        report["pair_ok"] = report["pair_ok"].fillna(False).astype(bool)
    return report


def summarize_mask_validation(report: pd.DataFrame) -> dict:
    """Summarize the output from :func:`validate_processed_mask_samples`."""
    required = {"partition", "image_ok", "mask_ok", "pair_ok", "mask_nonempty"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(
            "Validation report is missing columns: " + ", ".join(sorted(missing))
        )

    return {
        "checked_records": int(len(report)),
        "passed_pairs": int(report["pair_ok"].astype(bool).sum()),
        "failed_pairs": int((~report["pair_ok"].astype(bool)).sum()),
        "nonempty_masks": int(report["mask_nonempty"].astype(bool).sum()),
        "by_partition": {
            str(partition): {
                "checked": int(len(group)),
                "passed": int(group["pair_ok"].astype(bool).sum()),
                "failed": int((~group["pair_ok"].astype(bool)).sum()),
            }
            for partition, group in report.groupby("partition", sort=False)
        },
    }
