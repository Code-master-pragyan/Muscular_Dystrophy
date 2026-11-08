"""Configuration and provenance notes for Dataset B radiomics.

Extraction settings are taken from the two saved experiment JSON files.
The XGBoost values below describe the serialized booster configuration
inspected from Dataset_B_Radiomics_XGBoost.json. They are not a verified
record of the original Python training call or early-stopping procedure.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RadiomicsExtractionConfig:
    dataset_name: str = "Dataset B - FSHD Ultrasound"
    task: str = "Heckmatt grade classification"
    label_column: str = "heckmatt_grade"

    bin_width: int = 25
    mask_label: int = 1
    normalize: bool = False
    resampled_pixel_spacing: None = None
    image_type: str = "Original"

    feature_classes: Tuple[str, ...] = (
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
        "shape2D",
    )


@dataclass(frozen=True)
class SavedRadiomicsXGBoostConfig:
    """Observed properties of the saved booster, not training provenance."""

    objective: str = "multi:softprob"
    num_classes: int = 4
    num_features: int = 102
    boosting_rounds: int = 437

    booster: str = "gbtree"
    tree_method_in_saved_config: str = "auto"
    updater_in_saved_config: str = "grow_quantile_histmaker"
    learning_rate_in_saved_config: float = 0.3
    max_depth_in_saved_config: int = 6
    min_child_weight_in_saved_config: int = 1
    subsample_in_saved_config: float = 1.0
    colsample_bytree_in_saved_config: float = 1.0
    reg_lambda_in_saved_config: float = 1.0
    reg_alpha_in_saved_config: float = 0.0
    max_bin_in_saved_config: int = 256

    xgboost_version_in_saved_config: str = "3.4.1"


EXTRACTION_CONFIG = RadiomicsExtractionConfig()
SAVED_MODEL_CONFIG = SavedRadiomicsXGBoostConfig()

# Provenance limitations:
# - The original standalone Radiomics XGBoost training script/notebook cell
#   has not been located.
# - The saved booster configuration does not reveal the complete fit call,
#   data-loading code, validation protocol, class/sample weighting, random
#   seed used by the original wrapper, or early-stopping configuration.
# - Do not treat these saved-model properties as proof of the original
#   training script's arguments.
# - The project contains two non-identical extraction scripts. The exact
#   script that generated the archived feature CSVs has not been established.
