# Train-Only Model Robustness and Sensitivity Summary

## Objective

Consolidate the initial Logistic Regression robustness experiments before defining the final model and probability-calibration protocol.

The purpose is to identify consequential modeling assumptions and document provisional decisions.

## Data boundary

All experiments use only the frozen Train partition containing 1894 rows.

The reserved Calibration and Test partitions are not used for these analyses.

The original frozen split remains unchanged.

## Reference model

The reference model is Logistic Regression with:

- Ten approved predictors
- Fold-specific preprocessing inside a Pipeline
- Standardization of numeric predictors
- Binary Complains retained
- One-hot encoding for Age Group and Tariff Plan
- Numeric treatment of Charge Amount
- Regularization parameter C = 1.0

The reference model uses group-aware cross-validation within Train.

## 1. Complains ablation

Removing Complains produced the following mean changes:

| Metric | Change without Complains |
|---|---:|
| ROC-AUC | -0.0476 |
| Average Precision | -0.1881 |
| Brier Score | +0.0216 |
| Log Loss | +0.0579 |

Removing Complains worsened all four metrics.

For this paired comparison, groups were reconstructed using the nine remaining predictors.

Both model variants were evaluated on the same recomputed folds.

Therefore, these ablation results must not be directly compared with the original ten-feature CV fold results as if their validation partitions were identical.

Provisional decision: retain Complains.

Its temporal availability is supported by the dataset's general description, but individual complaint timestamps are unavailable.

Predictive importance does not establish a causal effect.

## 2. Charge Amount encoding

The experiment compared numeric encoding against one-hot encoding.

Mean changes for one-hot relative to numeric:

| Metric | Change with one-hot |
|---|---:|
| ROC-AUC | -0.0023 |
| Average Precision | -0.0017 |
| Brier Score | +0.0003 |
| Log Loss | +0.0024 |

The differences were small.

The Train partition contains eleven observed Charge Amount values, from 0 to 10.

The value 10 is rare and does not match the source's stated 0-to-9 description.

Values were preserved without modification.

Provisional decision: retain numeric encoding for Charge Amount.

This does not establish that the variable has a truly continuous economic interpretation.

## 3. Age Group encoding

The experiment compared one-hot encoding against numeric ordinal encoding.

Mean changes for ordinal relative to one-hot:

| Metric | Change with ordinal |
|---|---:|
| ROC-AUC | -0.0049 |
| Average Precision | -0.0112 |
| Brier Score | +0.0014 |
| Log Loss | +0.0051 |

One-hot encoding had better mean results across these metrics.

The original age-group codes represent categories rather than equally spaced age intervals.

Provisional decision: retain one-hot encoding for Age Group.

## 4. Regularization sensitivity

Three predefined Logistic Regression C values were compared.

| C | ROC-AUC | Average Precision | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
| 0.1 | 0.9232 | 0.7361 | 0.0763 | 0.2473 |
| 1.0 | 0.9288 | 0.7589 | 0.0715 | 0.2294 |
| 10.0 | 0.9294 | 0.7622 | 0.0711 | 0.2283 |

C = 0.1 produced worse mean results than C = 1.0.

C = 10.0 produced slightly better mean results than C = 1.0.

The differences between C = 1.0 and C = 10.0 were small.

Provisional decision: retain the initial C = 1.0 configuration.

This is a conservative development choice, not proof that C = 1.0 is optimal.

## Consolidated provisional decisions

| Component | Current decision |
|---|---|
| Complains | Retain |
| Charge Amount | Numeric encoding |
| Age Group | One-hot encoding |
| Regularization C | 1.0 |
| Customer Value | Exclude from prediction; reserve for decisions |
| Status | Remain excluded from primary model |
| Calibration | Not yet fitted |
| Test evaluation | Not yet performed |

These choices remain subject to the final model-freeze review.

## Scientific limitations

All results come from one dataset and internal cross-validation within its Train partition.

Cross-validation fold results are not independent experiments.

Small metric differences do not demonstrate statistical significance or practical importance.

Multiple development comparisons can increase the risk of overfitting decisions to Train.

The dataset has no verified unique customer identifier or chronological validation split.

The dataset does not record retention-intervention outcomes.

Future intervention benefits must therefore be simulated under explicit assumptions, not presented as observed causal effects.

## Reproduction

Run the four experiments:

    .\.venv\Scripts\python.exe -m experiments.complains_ablation_cv
    .\.venv\Scripts\python.exe -m experiments.charge_encoding_sensitivity
    .\.venv\Scripts\python.exe -m experiments.age_encoding_sensitivity
    .\.venv\Scripts\python.exe -m experiments.regularization_sensitivity

Run automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

## Detailed reports

- docs/complains_ablation.md
- docs/charge_encoding_sensitivity.md
- docs/age_encoding_sensitivity.md
- docs/regularization_sensitivity.md

## Next steps

1. Review and freeze the intended model configuration.
2. Define the probability-calibration protocol.
3. Fit the selected model on Train.
4. Use the reserved Calibration partition only for its predefined calibration role.
5. Preserve Test for final evaluation after development decisions are frozen.
