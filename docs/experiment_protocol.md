
# Experimental Protocol

## 1. Research Objective

The objective of this research is to investigate automated Heckmatt grade classification in Facioscapulohumeral Muscular Dystrophy (FSHD) using muscle ultrasound images.

The study investigates:

- CNN-based image classification.
- Handcrafted radiomics feature extraction.
- Group-level prediction and evaluation.
- CNN and radiomics feature integration.
- Explainability and interpretation of model predictions.

The current modelling experiments use Dataset B (FSHD ultrasound). Dataset A (MRI) is maintained as a separately audited dataset and is not combined with Dataset B.

## 2. Dataset Description

### 2.1 Dataset B — FSHD Ultrasound

The audited dataset contains:

| Description | Count |
|---|---:|
| Ultrasound images | 25,005 |
| Corresponding masks | 25,005 |
| Labelled image records | 4,727 |
| Unlabelled image records | 20,278 |
| Subjects | 109 |
| Subject–muscle–side groups | 1,468 |

Only records with available Heckmatt grades are included in supervised classification.

### 2.2 Heckmatt Classification

The classification task contains four grades:

| Class index | Heckmatt grade |
|---:|---|
| 0 | Grade 1 |
| 1 | Grade 2 |
| 2 | Grade 3 |
| 3 | Grade 4 |

The original grades are 1–4. Zero-based labels are used internally by the CNN classifier.

## 3. Subject-Level Data Splitting

A fixed subject-level split is used to prevent images from the same subject appearing in multiple partitions.

| Partition | Subjects | Image records | Groups |
|---|---:|---:|---:|
| Training | 76 | 3,342 | 1,029 |
| Validation | 17 | 741 | 231 |
| Testing | 16 | 644 | 208 |
| Total | 109 | 4,727 | 1,468 |

The approved split file is reused. The preprocessing code does not generate a new random split.

The group identifier is defined as:

`subject_id + "_" + muscle_code + "_" + side_code`

## 4. CNN Experiment

### 4.1 Architecture

The CNN baseline uses an ImageNet-pretrained ResNet18.

The original three-channel input convolution is converted to a single-channel convolution. Its pretrained weights are initialized by averaging the three RGB input-channel weights.

The final fully connected layer is replaced with a four-class classification layer.

The pooled feature representation contains 512 features per image.

### 4.2 Image Preprocessing

The CNN preprocessing pipeline uses:

1. Grayscale image loading.
2. Resize to 224 × 224 pixels.
3. Conversion to a PyTorch tensor.
4. Normalization using mean 0.485 and standard deviation 0.229.

The CNN dataset loader uses image files only; it does not use the segmentation masks during classification.

### 4.3 Training Configuration

The current CNN configuration records:

| Parameter | Value |
|---|---:|
| Architecture | ResNet18 |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Maximum epochs | 30 |
| Early stopping patience | 7 |
| Random seed | 42 |
| Number of classes | 4 |

The training function receives the loss function and optimizer from its caller.

Model selection is based on validation group-level macro-F1.

### 4.4 Group-Level Evaluation

Image probabilities are aggregated within each subject–muscle–side group.

For each group:

1. Calculate the mean probability for each of the four classes.
2. Select the class with the highest mean probability.
3. Compare the resulting group prediction with the group's ground-truth grade.

## 5. Radiomics Experiment

### 5.1 Feature Extraction

Radiomics features are extracted using PyRadiomics.

The documented extraction configuration is:

| Parameter | Setting |
|---|---|
| Image type | Original |
| Bin width | 25 |
| Mask label | 1 |
| Normalize | False |
| Resampled pixel spacing | None |

The enabled feature classes are:

- First Order
- GLCM
- GLRLM
- GLSZM
- GLDM
- NGTDM
- Shape2D

The audited feature tables contain 102 radiomics features.

### 5.2 Mask Handling

The original image-mask audit found binary masks encoded using pixel values 0 and 255.

The processed binary masks use values 0 and 1.

The current reusable extraction implementation accepts these two binary encodings and converts foreground pixels to label 1 before extraction.

Empty masks and unexpected mask values are rejected.

### 5.3 Group-Level Feature Aggregation

The repository includes a utility that aggregates image-level radiomics features by:

- Subject
- Muscle
- Side

The arithmetic mean is calculated for each selected feature within a group.

The output contains one row per group, the corresponding Heckmatt grade, the image count and the aggregated feature values.

This utility is a reusable implementation. Its presence does not establish that this exact aggregation procedure was used to generate the historical XGBoost results.

## 6. Evaluation Metrics

The classification experiments use:

- Accuracy
- Macro F1-score
- Balanced accuracy
- Confusion matrix
- Class-wise classification report

Image-level and group-level metrics are reported separately.

Model comparisons must use the same test subjects and the same evaluation unit.

## 7. Statistical Analysis

The hybrid-versus-CNN comparison uses paired subject-level bootstrap resampling.

The recorded analysis used:

- 2,000 bootstrap iterations.
- 16 test subject clusters.
- Random seed 42.
- Macro-F1 difference between the hybrid model and CNN.

Confidence intervals are interpreted alongside the observed metric differences. An interval containing zero does not provide clear evidence of an improvement.

## 8. Reproducibility and Limitations

The following limitations must be considered:

1. The original standalone radiomics XGBoost training script was not recovered.
2. The saved XGBoost configuration describes observed booster properties, not the complete original training call.
3. Two non-identical radiomics extraction scripts were recovered. The exact script used to generate the archived feature CSV files has not been established.
4. The hybrid-model audit found 54 selected features, all CNN features, with zero selected radiomics features.
5. The audited hybrid model showed no statistically clear macro-F1 improvement over the CNN baseline.
6. Grade 4 has relatively low support, which increases uncertainty for that class.
7. Available PNG spacing metadata is not verified physical millimetre spacing. Shape measurements must not be interpreted as calibrated physical dimensions.

The implementation should distinguish verified historical experimental results from newly written reusable utilities.

## 9. Research Scope

This project is an experimental research framework for FSHD ultrasound image classification.

It is not a clinically validated diagnostic system and should not be presented as a replacement for professional medical assessment.
