"""
Radiomics utilities for Dataset B FSHD ultrasound analysis.

The package contains configurable feature extraction and prediction-evaluation
utilities. The original radiomics XGBoost training script was not recovered,
so this package does not claim to reproduce that training procedure.
"""

from .evaluate import (
    aggregate_image_predictions_to_groups,
    calculate_classification_metrics,
    evaluate_group_predictions,
    evaluate_image_predictions,
)

__all__ = [
    "aggregate_image_predictions_to_groups",
    "calculate_classification_metrics",
    "evaluate_group_predictions",
    "evaluate_image_predictions",
]
