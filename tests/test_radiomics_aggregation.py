"""Unit tests for radiomics group-level aggregation."""

import pandas as pd
import pytest

from src.radiomics.aggregation import aggregate_radiomics_by_group


def sample_features():
    return pd.DataFrame(
        {
            "subject_id": ["S1", "S1", "S1", "S2"],
            "muscle_code": [1, 1, 1, 2],
            "side_code": [1, 1, 1, 2],
            "heckmatt_grade": [2, 2, 2, 3],
            "original_firstorder_Mean": [1.0, 3.0, 5.0, 8.0],
            "original_glcm_Contrast": [2.0, 4.0, 6.0, 10.0],
        }
    )


def test_aggregation_returns_one_row_per_subject_muscle_side():
    result = aggregate_radiomics_by_group(sample_features())

    assert len(result) == 2
    assert set(result["image_count"]) == {1, 3}


def test_aggregation_calculates_arithmetic_mean():
    result = aggregate_radiomics_by_group(sample_features())
    group = result[
        (result["subject_id"] == "S1")
        & (result["muscle_code"] == 1)
        & (result["side_code"] == 1)
    ].iloc[0]

    assert group["original_firstorder_Mean"] == pytest.approx(3.0)
    assert group["original_glcm_Contrast"] == pytest.approx(4.0)
    assert group["heckmatt_grade"] == 2


def test_inconsistent_labels_within_group_are_rejected():
    df = sample_features()
    df.loc[1, "heckmatt_grade"] = 3

    with pytest.raises(ValueError, match="inconsistent"):
        aggregate_radiomics_by_group(df)


def test_invalid_grade_is_rejected():
    df = sample_features()
    df.loc[0, "heckmatt_grade"] = 5

    with pytest.raises(ValueError, match="grades must be integers"):
        aggregate_radiomics_by_group(df)


def test_non_numeric_feature_is_rejected_when_explicitly_selected():
    df = sample_features()
    df["bad_feature"] = ["a", "b", "c", "d"]

    with pytest.raises(ValueError, match="must be numeric"):
        aggregate_radiomics_by_group(
            df,
            feature_columns=["bad_feature"],
        )


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="is empty"):
        aggregate_radiomics_by_group(sample_features().iloc[0:0])
