# Hybrid CNN and Radiomics Framework for FSHD Ultrasound Classification

An experimental research framework for automated classification of Facioscapulohumeral Muscular Dystrophy (FSHD) ultrasound images using deep learning, handcrafted radiomics, feature fusion and explainable AI.

## 1. Project Overview

The objective is to classify muscle ultrasound images into four Heckmatt grades (Grade 1–4).

The project investigates:

- CNN-based image classification using ResNet18.
- Handcrafted radiomics feature extraction using PyRadiomics.
- Subject–muscle–side level aggregation and evaluation.
- CNN embedding and radiomics feature integration.
- XGBoost-based hybrid classification.
- Feature importance and SHAP explainability.
- Statistical comparison of model performance.

**Research scope:** Dataset B (FSHD ultrasound) is used for the current modelling experiments. Dataset A (MRI) has been audited separately and is not combined with Dataset B.

## 2. Dataset Information

### Dataset B — FSHD Ultrasound

| Description | Count |
|---|---:|
| Original ultrasound images | 25,005 |
| Corresponding masks | 25,005 |
| Labelled image records | 4,727 |
| Unlabelled records excluded | 20,278 |
| Subjects | 109 |
| Subject–muscle–side groups | 1,468 |

Only labelled records with valid Heckmatt grades are used for supervised classification.

### Approved Subject-Level Split

| Partition | Subjects | Image records | Groups |
|---|---:|---:|---:|
| Training | 76 | 3,342 | 1,029 |
| Validation | 17 | 741 | 231 |
| Testing | 16 | 644 | 208 |
| **Total** | **109** | **4,727** | **1,468** |

The approved subject-level split is reused across experiments to prevent subject overlap between partitions.

### Dataset A — MRI

The Benchmark Diagnostic MRI and Medical Imaging Dataset was audited separately. It is not used in the current FSHD ultrasound modelling pipeline.

## 3. Methodology

### 3.1 Preprocessing

- Grayscale ultrasound image loading.
- Metadata and label validation.
- Image-mask consistency checks.
- Subject-level partition validation.
- CNN image resizing to 224 × 224 pixels.
- Tensor conversion and normalization.

The CNN uses images only; segmentation masks are not CNN inputs.

### 3.2 CNN Classification

The baseline model uses ImageNet-pretrained ResNet18.

| Parameter | Setting |
|---|---|
| Architecture | ResNet18 |
| Input channels | 1 (grayscale) |
| Image size | 224 × 224 |
| Output classes | 4 |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Maximum epochs | 30 |
| Early stopping patience | 7 |
| Random seed | 42 |

The pretrained RGB input convolution weights are converted to single-channel initialization by averaging the RGB channel weights.

The pooled CNN representation contains 512 features per image.

### 3.3 Radiomics Extraction

Radiomics features are extracted using PyRadiomics.

| Parameter | Setting |
|---|---|
| Image type | Original |
| Bin width | 25 |
| Mask label | 1 |
| Normalize | False |
| Resampling | Not enabled |
| Feature classes | First Order, GLCM, GLRLM, GLSZM, GLDM, NGTDM, Shape2D |

The audited feature tables contain 102 radiomics features.

### 3.4 Feature Fusion

The image-level fusion representation contains:

- 512 CNN embedding features.
- 102 radiomics features.
- 614 combined input features.

The hybrid analysis includes feature selection, XGBoost classification, explainability and statistical comparison.

**Important observed result:** The audited selected hybrid booster contains 54 selected features, all from the CNN modality and zero from radiomics. The modality-level SHAP summary attributes 100% to CNN and 0% to radiomics. Therefore, the archived hybrid result should not be presented as evidence of a selected radiomics contribution.

## 4. Key Experimental Findings

The archived comparison reported:

- Hybrid macro-F1 improvement over CNN: approximately 0.54 percentage points.
- Hybrid balanced accuracy difference relative to CNN: approximately -1.33 percentage points.
- Paired subject-level bootstrap 95% confidence interval for the hybrid-minus-CNN macro-F1 difference: [-0.0453, 0.0447].

The confidence interval includes zero. Therefore, the observed macro-F1 difference does not establish a statistically clear improvement over the CNN baseline.

Results are interpreted using the same held-out subject groups and the documented evaluation methodology.

## 5. Repository Structure

```text
Muscular_Dystrophy/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── notebooks/
│   ├── README.md
│   └── Mascular_Dystrophy_Research.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── manifest.py
│   │   ├── split.py
│   │   ├── mask_validation.py
│   │   └── validation.py
│   │
│   ├── cnn/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── loaders.py
│   │   ├── optimization.py
│   │   ├── artifacts.py
│   │   ├── checkpoint.py
│   │   ├── results.py
│   │   ├── experiment.py
│   │   ├── training_plots.py
│   │   ├── predictions.py
│   │   ├── image_results.py
│   │   └── embeddings.py
│   │
│   ├── radiomics/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── extraction.py
│   │   ├── aggregation.py
│   │   ├── evaluate.py
│   │   ├── quality.py
│   │   ├── feature_analysis.py
│   │   ├── validation.py
│   │   ├── audit.py
│   │   └── radiomics_README.md
│   │
│   └── fusion/
│       ├── merge.py
│       ├── aggregation.py
│       ├── train.py
│       ├── evaluate.py
│       ├── explainability.py
│       ├── statistics.py
│       └── figures.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_radiomics_aggregation.py
│
├── docs/
│   └── experiment_protocol.md
│
├── results/
│   ├── cnn/
│   ├── radiomics/
│   └── hybrid/
│
├── figures/
│   ├── README.md
│   └── research figures
│
├── README.md
└── requirements.txt
```

The original notebook records the Colab experimental workflow. The `src/` directory contains reusable modular Python implementations.

## 6. Installation

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

Install dependencies:

```bash
pip install -r requirements.txt
```

Install a PyTorch and Torchvision build compatible with your hardware and CUDA environment where required.

## 7. Testing

Run the unit tests from the project root:

```bash
python -m pytest -v
```

Check Python syntax:

```bash
python -m compileall -q src tests
```

### GitHub Actions

The repository includes an automated GitHub Actions workflow.

The workflow runs on pushes to `main`, pull requests targeting `main`, and manual execution.

It performs:

1. Python syntax compilation.
2. Preprocessing unit tests.
3. Radiomics aggregation unit tests.

The current test suite contains 13 tests, which passed in the latest verified successful GitHub Actions run.

These checks do not represent a complete end-to-end training run.

## 8. Results and Artifacts

The `results/` directory contains selected archived experimental outputs, including:

- CNN image-level and group-level metrics.
- Radiomics evaluation results.
- Hybrid model predictions and evaluation.
- Feature selection summaries.
- Statistical comparison tables.

The `figures/` directory contains research figures generated from the experiments:

- CNN training and validation curves.
- Group-level confusion matrices.
- Model performance confidence intervals.
- Radiomics and hybrid explainability figures.

Only actual outputs generated by the experiments should be presented as experimental results.

## 9. Reproducibility and Limitations

1. The original standalone radiomics XGBoost training script has not been recovered. The saved booster configuration does not establish the complete original training procedure.
2. Two non-identical radiomics extraction scripts were recovered. The exact script that generated the archived feature CSV files has not been established.
3. Available image spacing metadata is not verified physical millimetre spacing. Shape measurements should not be interpreted as calibrated physical dimensions.
4. Heckmatt Grade 4 has relatively low sample support, increasing uncertainty for that class.
5. The approved subject-level split must be preserved for fair model comparison.
6. Some modular utilities were reconstructed from the exported notebook workflow where original source cells were unavailable. They should not all be described as exact historical reproductions.
7. The current framework is experimental and has not been clinically validated.

## 10. Research Integrity

The project distinguishes between:

- Original Colab experimental workflow.
- Reusable modular Python implementations.
- Observed saved-model properties.
- Verified experimental findings.
- Reconstructed or newly written utilities.

The presence of radiomics features in a combined input table does not establish that radiomics contributed to the selected hybrid classifier.

No unverified results, procedures or clinical performance claims should be presented as established findings.

## 11. Repository

**GitHub:** https://github.com/Code-master-pragyan/Muscular_Dystrophy

**Project:** Hybrid CNN and Radiomics Framework for FSHD Ultrasound Classification

**Status:** Research and experimental development.
