
# Research Notebooks

This directory stores Jupyter and Google Colab notebooks used during the FSHD ultrasound research project.

## Recommended Organization

| Notebook | Purpose |
|---|---|
| `01_dataset_audit.ipynb` | Dataset inspection, metadata checks and label validation |
| `02_preprocessing.ipynb` | Manifest preparation, approved split integration and preprocessing checks |
| `03_cnn_training.ipynb` | ResNet18 training and validation |
| `04_cnn_evaluation.ipynb` | CNN image-level and group-level evaluation |
| `05_radiomics_extraction.ipynb` | PyRadiomics feature extraction |
| `06_radiomics_evaluation.ipynb` | Radiomics prediction evaluation |
| `07_hybrid_analysis.ipynb` | CNN and radiomics feature integration analysis |
| `08_explainability.ipynb` | Feature selection and explainability analysis |

## Notebook Guidelines

- Keep notebooks focused on one research task.
- Use relative project paths where possible.
- Record random seeds and experiment configurations.
- Preserve the approved subject-level split.
- Avoid committing datasets, model checkpoints or large generated outputs.
- Move reusable functions into the `src/` package instead of duplicating them across notebooks.
- Keep results reproducible and distinguish exploratory experiments from reported results.

## Dataset Scope

The current modelling notebooks should use Dataset B (FSHD ultrasound).

Dataset A (MRI) is maintained separately and must not be combined with Dataset B in the current modelling workflow.

## Reproducibility

Each notebook should include:

1. Research objective.
2. Required input files.
3. Configuration and dependencies.
4. Main execution steps.
5. Output files and evaluation metrics.
6. Relevant limitations.
