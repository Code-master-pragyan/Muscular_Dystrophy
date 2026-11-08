# Radiomics Module — Dataset B (FSHD Ultrasound)

This module contains reusable radiomics feature-extraction and prediction-evaluation
utilities for the Dataset B FSHD ultrasound workflow.

## Scope

- Dataset: Dataset B — FSHD ultrasound.
- Task: Heckmatt grade classification (grades 1–4).
- Feature extraction: PyRadiomics, using the configured 2D ultrasound settings.
- Evaluation: image-level metrics and group-level evaluation after averaging
  image probabilities within each `group_id`.

Dataset A (MRI) is audited separately and is not combined with Dataset B in
this workflow.

## Files

| File | Purpose |
|---|---|
| `config.py` | Records the radiomics extraction settings and the observed saved XGBoost model configuration. |
| `extraction.py` | Extracts radiomics features from images and masks and writes feature/error CSV files. |
| `evaluate.py` | Aggregates image probabilities to groups and calculates classification metrics. |
| `__init__.py` | Exposes the main evaluation helpers from the package. |

## Feature extraction settings

The documented extraction settings are:

- `binWidth = 25`
- `label = 1`
- `normalize = False`
- `resampledPixelSpacing = None`
- Image type: Original
- Feature classes: firstorder, glcm, glrlm, glszm, gldm, ngtdm, shape2D

The audited feature tables contain 102 radiomics features in addition to
metadata columns. The feature tables do not establish which of the two
recovered extraction scripts generated them.

## Evaluation convention

The group-level evaluation helper:

1. Groups image predictions by `group_id`.
2. Averages the four grade probabilities (`probability_G1` through
   `probability_G4`) within each group.
3. Assigns the group prediction using the largest mean probability.
4. Calculates accuracy, macro-F1, balanced accuracy, and a confusion matrix.

Ground-truth Heckmatt grade is expected to be consistent within each group.

## Model provenance limitation

The saved radiomics XGBoost artifact was inspected, but its original training
script was not recovered. The saved booster configuration records 102 input
features, four classes, 437 boosting rounds, and XGBoost 3.4.1. These saved
properties do not establish the complete original training call, preprocessing,
validation strategy, or early-stopping procedure.

Therefore, this repository documents the observed model configuration but
does not claim to reproduce the original radiomics model-training procedure.
Do not create or cite a reconstructed `train.py` as the original experiment.

## Important interpretation notes

- The 102-feature count describes the audited feature tables; it does not prove
  that all features contributed to the final hybrid classifier.
- The separate hybrid-model audit found that its selected features were CNN
  features, with no selected radiomics features. Do not claim a measured
  radiomics contribution to that hybrid model.
- Image-level and group-level metrics are different evaluation units and must
  not be compared as if they were the same.
- The available PNG metadata spacing is not verified physical millimetre
  spacing. Shape measurements should not be interpreted as calibrated
  physical dimensions without verified spacing metadata.

## Dependencies

The extraction and evaluation modules use:

- Python
- NumPy
- pandas
- scikit-learn
- SimpleITK
- PyRadiomics

Install compatible versions in the project environment before running
feature extraction. The exact original extraction environment is not fully
established by the recovered artifacts.
