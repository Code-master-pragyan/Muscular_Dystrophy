# Hybrid CNN and Radiomics Framework for FSHD Ultrasound Classification

A research project investigating automated Heckmatt grade classification in Facioscapulohumeral Muscular Dystrophy (FSHD) using ultrasound images, deep learning and radiomics.

## 1. Project Overview

This project investigates an automated approach for classifying muscle ultrasound images into four Heckmatt grades (Grade 1–4).

The research focuses on:

- Image-based classification using a convolutional neural network (CNN).
- Handcrafted radiomics feature extraction using PyRadiomics.
- Group-level evaluation at the subject–muscle–side level.
- Investigation of CNN and radiomics feature integration.
- Explainability and careful interpretation of model performance.

**Research scope:** The current modelling experiments use Dataset B (FSHD ultrasound). Dataset A (MRI) has been audited separately and is not combined with Dataset B or used in the current modelling pipeline.

## 2. Dataset Information

### Dataset B — FSHD Ultrasound

The ultrasound dataset contains:

| Description | Count |
|---|---:|
| Original ultrasound images | 25,005 |
| Corresponding masks | 25,005 |
| Labelled image records used | 4,727 |
| Unlabelled records excluded | 20,278 |
| Subjects | 109 |
| Subject–muscle–side groups | 1,468 |

Only labelled records with valid Heckmatt grades are used in supervised classification.

### Subject-Level Data Split

| Partition | Subjects | Image records | Groups |
|---|---:|---:|---:|
| Training | 76 | 3,342 | 1,029 |
| Validation | 17 | 741 | 231 |
| Testing | 16 | 644 | 208 |
| **Total** | **109** | **4,727** | **1,468** |

The approved subject-level split is reused throughout the experiments to prevent subject overlap between partitions.

### Dataset A — MRI

The Benchmark Diagnostic MRI and Medical Imaging Dataset has been audited separately. It is not combined with the FSHD ultrasound dataset in the current experiments.

## 3. Methodology

### 3.1 Image Preprocessing

The CNN preprocessing pipeline uses:

- Grayscale ultrasound images.
- Resizing to 224 × 224 pixels.
- Conversion to tensors.
- Normalization using mean 0.485 and standard deviation 0.229.

### 3.2 CNN Classification

The CNN baseline uses an ImageNet-pretrained ResNet18 architecture.

Key implementation details:

- Single-channel grayscale input.
- Pretrained RGB convolution weights converted to grayscale initialization by averaging the input-channel weights.
- Four output classes corresponding to Heckmatt Grades 1–4.
- Training and model selection using validation group-level macro-F1.

The pooled ResNet18 representation contains 512 features per image.

### 3.3 Radiomics Feature Extraction

Radiomics features are extracted using PyRadiomics.

The documented extraction configuration includes:

- Bin width: 25.
- Mask label: 1.
- Normalization: disabled.
- Image type: Original.
- Feature classes: First Order, GLCM, GLRLM, GLSZM, GLDM, NGTDM and Shape2D.

The audited radiomics feature tables contain 102 radiomics features.

### 3.4 Group-Level Analysis

Images belonging to the same subject–muscle–side group are aggregated for group-level analysis.

The radiomics aggregation utility calculates the arithmetic mean of image-level features within each group.

Image-level and group-level evaluation results are treated as separate evaluation units.

## 4. Repository Structure

```text
Muscular_Dystrophy/
│
├── src/
│   ├── cnn/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── validation.py
│   │
│   └── radiomics/
│       ├── __init__.py
│       ├── config.py
│       ├── extraction.py
│       ├── aggregation.py
│       ├── evaluate.py
│       └── radiomics_README.md
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_radiomics_aggregation.py
│
├── README.md
└── requirements.txt
```

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/Code-master-pragyan/Muscular_Dystrophy.git

cd Muscular_Dystrophy
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Install a compatible PyTorch and Torchvision build for your hardware and CUDA environment.

## 6. Running Tests

Run the existing test suite from the repository root:

```bash
python -m pytest tests/ -v
```

The tests cover preprocessing validation and radiomics group aggregation.

## 7. Research Limitations and Reproducibility

The following limitations are important for interpreting the experiments:

1. The original standalone radiomics XGBoost training script has not been recovered. The saved booster configuration does not establish the complete original training procedure.
2. Two non-identical radiomics extraction scripts were recovered. The exact script that generated the archived feature CSV files has not been established.
3. The available image metadata spacing is not verified physical millimetre spacing. Shape measurements should not be interpreted as calibrated physical dimensions.
4. Heckmatt Grade 4 has relatively low sample support, resulting in greater uncertainty for that class.
5. CNN and radiomics results must be compared using the same evaluation unit and test groups.
6. The current implementation is an experimental research framework, not a clinically validated diagnostic system.

## 8. Research Integrity

The project distinguishes between implemented methods, observed saved-model properties and independently verified experimental findings.

In particular, the presence of radiomics features in an input feature table does not establish that they contributed to the final hybrid classifier.

No unverified architectural components, training procedures or clinical performance claims are presented as established findings.

## 9. Repository

**GitHub:** https://github.com/Code-master-pragyan/Muscular_Dystrophy

**Project:** FSHD Ultrasound-Based Heckmatt Grade Classification

**Status:** Research and experimental development.
