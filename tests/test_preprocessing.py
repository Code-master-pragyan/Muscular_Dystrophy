"""Unit tests for the preprocessing validation utilities.

Run from the repository root with:
    python -m pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import pytest

from src.preprocessing.validation import validate_partition_dataframes


def make_partition(partition_name, subject_id, filename):
    """Create a minimal valid one-record partition for testing."""
    return pd.DataFrame(
        [
            {
                "filename": filename,
                "image_path": f"/not_checked/{filename}",
                "subject_id": subject_id,
                "muscle_code": 1,
                "side_code": 1,
                "heckmatt_grade": 2,
                "label": 1,
                "group_id": f"{subject_id}_1_1",
                "partition": partition_name,
            }
        ]
    )


def make_valid_partitions():
    return {
        "train": make_partition("train", "subject_train", "train_001.png"),
        "validation": make_partition(
            "validation", "subject_validation", "validation_001.png"
        ),
        "test": make_partition("test", "subject_test", "test_001.png"),
    }


def test_valid_partitions_pass_validation():
    partitions = make_valid_partitions()

    report = validate_partition_dataframes(
        partitions,
        check_image_files=False,
        expected_counts=None,
    )

    assert report["train"]["records"] == 1
    assert report["validation"]["records"] == 1
    assert report["test"]["records"] == 1
    assert report["total"]["records"] == 3
    assert report["total"]["subjects"] == 3


def test_subject_overlap_is_rejected():
    partitions = make_valid_partitions()
    partitions["test"].loc[0, "subject_id"] = "subject_train"

    with pytest.raises(ValueError, match="Subject leakage"):
        validate_partition_dataframes(
            partitions,
            check_image_files=False,
            expected_counts=None,
        )


def test_label_must_equal_grade_minus_one():
    partitions = make_valid_partitions()
    partitions["train"].loc[0, "label"] = 0

    with pytest.raises(ValueError, match="label must equal"):
        validate_partition_dataframes(
            partitions,
            check_image_files=False,
            expected_counts=None,
        )


def test_invalid_heckmatt_grade_is_rejected():
    partitions = make_valid_partitions()
    partitions["validation"].loc[0, "heckmatt_grade"] = 5
    partitions["validation"].loc[0, "label"] = 4

    with pytest.raises(ValueError, match="invalid Heckmatt grades"):
        validate_partition_dataframes(
            partitions,
            check_image_files=False,
            expected_counts=None,
        )


def test_duplicate_filename_across_partitions_is_rejected():
    partitions = make_valid_partitions()
    partitions["test"].loc[0, "filename"] = "train_001.png"

    with pytest.raises(ValueError, match="Duplicate filenames"):
        validate_partition_dataframes(
            partitions,
            check_image_files=False,
            expected_counts=None,
        )


def test_missing_required_column_is_rejected():
    partitions = make_valid_partitions()
    partitions["train"] = partitions["train"].drop(columns=["group_id"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_partition_dataframes(
            partitions,
            check_image_files=False,
            expected_counts=None,
        )


def test_wrong_partition_value_is_rejected():
    partitions = make_valid_partitions()
    partitions["test"].loc[0, "partition"] = "train"

    with pytest.raises(ValueError, match="incorrect partition value"):
        validate_partition_dataframes(
            partitions,
            check_image_files=False,
            expected_counts=None,
        )
