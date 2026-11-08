"""Configuration for the Dataset B ResNet18 CNN baseline.

Values are based on the original Step_2_Dataset_Audit.ipynb CNN experiment.
This module centralizes experiment settings; it does not change the model
architecture, data split, or evaluation protocol.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CNNConfig:
    # Image preprocessing
    image_size: int = 224
    grayscale: bool = True
    normalize_mean: float = 0.485
    normalize_std: float = 0.229

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    early_stopping_patience: int = 7
    random_seed: int = 42

    # Classification
    num_classes: int = 4
    class_names: Tuple[str, ...] = ("Grade 1", "Grade 2", "Grade 3", "Grade 4")

    # Dataset split sizes from the audited subject-level split
    expected_train_images: int = 3342
    expected_validation_images: int = 741
    expected_test_images: int = 644
    expected_total_images: int = 4727
    expected_test_groups: int = 208

    # Model selection and group-level evaluation
    selection_metric: str = "validation_group_macro_f1"
    group_aggregation: str = "mean_probability"


CONFIG = CNNConfig()

# Original labels are grades 1–4; the model uses zero-based class indices.
GRADE_TO_CLASS: Dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

CLASS_TO_GRADE: Dict[int, int] = {
    class_index: grade for grade, class_index in GRADE_TO_CLASS.items()
}
