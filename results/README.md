
# Results

This directory stores generated outputs from the Dataset B (FSHD ultrasound) experiments.

## Recommended Structure

```text
results/
├── cnn/
│   ├── image_level_metrics.csv
│   ├── group_level_metrics.csv
│   └── group_predictions.csv
├── radiomics/
│   ├── image_level_metrics.csv
│   ├── group_level_metrics.csv
│   └── group_predictions.csv
└── hybrid/
    ├── group_level_metrics.csv
    ├── group_predictions.csv
    └── feature_selection_summary.csv
```

These are suggested folders and filenames. Add files only when corresponding verified outputs exist.

## Reporting Rules

- Keep image-level and group-level metrics separate.
- Use identical held-out test subjects for model comparisons.
- Record the class order: Heckmatt Grades 1, 2, 3 and 4.
- Record the number of images, groups and subjects evaluated.
- Document experiment configurations and random seeds.
- Preserve raw predictions separately from summarized metrics.
- Do not overwrite original experimental outputs.

## Research Interpretation

The original radiomics XGBoost training script has not been recovered.

The hybrid-model audit found that all selected features were CNN features, with zero selected radiomics features. Do not claim a verified radiomics contribution to the archived hybrid model.

Do not report synthetic or unverified values as experimental results.
