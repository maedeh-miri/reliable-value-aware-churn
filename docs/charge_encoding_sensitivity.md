# Charge Amount Encoding Sensitivity

## Objective

Examine whether the initial Logistic Regression baseline is sensitive to treating Charge Amount as numeric versus one-hot encoded.

This is a preprocessing sensitivity analysis, not final model selection.

## Data boundary

Only the frozen v1 Train partition is used for model fitting and internal validation.

Calibration and Test remain excluded from model fitting, tuning, and sensitivity comparisons.

The frozen v1 manifest and metadata are not modified.

## Feature semantics

The UCI dataset documentation describes Charge Amount as an ordinal attribute with values from 0 to 9.

The original data audit identified seven rows with value 10 in the complete dataset.

Within the frozen Train partition, four rows have Charge Amount equal to 10.

These values are preserved without automatic correction or deletion.

## Comparison design

Two Logistic Regression pipelines are compared.

Numeric model:

- Charge Amount is included among the standardized numeric predictors.
- StandardScaler is fitted within each training fold.

One-hot model:

- Charge Amount is removed from the numeric transformer.
- OneHotEncoder handles it as a categorical predictor.
- Unknown categories are handled with handle_unknown="ignore".

All other predictors and preprocessing decisions remain unchanged.

Both models use the same initial Logistic Regression configuration:

- L2 regularization
- C = 1.0
- Solver = lbfgs
- Maximum iterations = 1000

## Cross-validation protocol

- Method: StratifiedGroupKFold
- Internal folds: 5
- Shuffle: True
- Random state: 42
- Data: frozen v1 Train only
- Group definition: exact equality across the ten approved predictors

The same fit and validation rows are used for both encoding variants within each fold.

No predictor-pattern group overlaps between fit and validation.

## Rare-category diagnostics

Charge Amount has eleven observed values in Train, from 0 through 10.

Value 10 occurs in only four Train rows.

Across the five internal folds, the value 10 appears between two and four times in each fit portion.

No Charge Amount category was present in validation while absent from the corresponding fit portion in this experiment.

Rare categories nevertheless provide limited information for estimating individual category effects.

## Mean cross-validation results

| Metric | Numeric | One-Hot | Delta |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.9288 | 0.9265 | -0.0023 |
| Average Precision | 0.7589 | 0.7572 | -0.0017 |
| Brier Score | 0.0715 | 0.0718 | +0.0003 |
| Log Loss | 0.2294 | 0.2318 | +0.0024 |

Delta is defined as the one-hot metric minus the numeric metric.

Values are rounded to four decimal places.

For ROC-AUC and Average Precision, higher is better.

For Brier Score and Log Loss, lower is better.

## Interpretation

The two encodings produced similar predictive performance in the five internal folds.

Numeric encoding had slightly better mean ROC-AUC, Average Precision, Brier Score, and Log Loss.

The differences were small compared with those observed in the Complains ablation experiment.

These results suggest limited sensitivity to Charge Amount encoding under the tested model configuration and data split.

Small observed differences do not establish statistical equivalence.

## Provisional decision

Retain the numeric representation in the initial Logistic Regression baseline.

This choice preserves a simpler feature representation and is not contradicted by the internal cross-validation results.

The decision does not establish that the numeric representation is universally optimal.

## Limitations

The source documentation describes Charge Amount as ordinal, and the observed value 10 is outside its stated range.

The exact spacing and business meaning of Charge Amount categories are uncertain.

Rare categories may produce unstable category-specific estimates.

Only one fixed regularization configuration was compared.

The data split is not chronological and has no verified customer identifiers.

The results are development-stage Train cross-validation estimates, not final Test performance.

No probability calibration, intervention-effect estimation, or retention-policy optimization has been performed.

## Reproduction

Run from the repository root with the documented environment and verified raw dataset:

    .\.venv\Scripts\python.exe -m experiments.charge_encoding_sensitivity

Run automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

## References

UCI Iranian Churn Dataset, ID 563:
https://archive.ics.uci.edu/dataset/563/iranian%2Bchurn%2Bdataset

## Next steps

- Examine sensitivity to Age Group encoding.
- Study reasonable regularization settings within Train.
- Establish a probability-calibration protocol before using Calibration.
- Preserve Test for final evaluation.
