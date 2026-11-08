
"""
Radiomics utilities for Dataset B FSHD ultrasound analysis.

The original radiomics XGBoost training script was not recovered.
This package does not claim to reproduce that training procedure.
"""

from .aggregation import aggregate_radiomics_by_group

from .evaluate import (
    aggregate_image_predictions_to_groups,
    calculate_classification_metrics,
    evaluate_group_predictions,
    evaluate_image_predictions,
)

__all__ = [
    "aggregate_radiomics_by_group",
    "aggregate_image_predictions_to_groups",
    "calculate_classification_metrics",
    "evaluate_group_predictions",
    "evaluate_image_predictions",
]
