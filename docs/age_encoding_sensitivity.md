# Age Group Encoding Sensitivity

## Objective

Examine whether the initial Logistic Regression baseline is sensitive to representing Age Group with one-hot encoding versus standardized ordinal-numeric codes.

This is a preprocessing sensitivity analysis, not final model selection.

## Data boundary

Only the frozen v1 Train partition is used.

Calibration and Test remain excluded from model fitting, tuning, and sensitivity comparisons.

The frozen v1 manifest and metadata are not modified.

## Feature semantics

Age Group is an ordinal-coded predictor with values from 1 to 5.

The observed dataset maps these codes to ages 15, 25, 30, 45, and 55, respectively.

The gaps between these represented ages are not equal.

The separate Age predictor remains excluded from the primary model because it is deterministically redundant with Age Group in this dataset.

## Comparison design

Two Logistic Regression pipelines are compared.

One-hot model:

- Age Group is one-hot encoded.
- Each observed age group receives a separate indicator column.

Ordinal-numeric model:

- Age Group is included among standardized numeric predictors.
- A linear contribution of the ordinal code to log-odds is assumed.

All other predictors and preprocessing choices remain unchanged.

Both models use:

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

Both encoding variants use identical fit and validation rows within each fold.

Preprocessing is fitted exclusively on each fold's fit portion.

## Mean cross-validation results

| Metric | One-Hot | Ordinal Numeric | Delta |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.9288 | 0.9239 | -0.0049 |
| Average Precision | 0.7589 | 0.7478 | -0.0112 |
| Brier Score | 0.0715 | 0.0729 | +0.0014 |
| Log Loss | 0.2294 | 0.2345 | +0.0051 |

Delta is defined as the ordinal-numeric metric minus the one-hot metric.

Values are rounded to four decimal places.

For ROC-AUC and Average Precision, higher is better.

For Brier Score and Log Loss, lower is better.

## Fold-level observations

One-hot encoding achieved higher ROC-AUC and Average Precision in all five internal folds.

Log Loss was lower for one-hot encoding in all five folds.

Brier Score was lower for one-hot encoding in four of five folds.

The magnitude of the differences was relatively small.

## Interpretation

The results show a modest and directionally consistent advantage for one-hot encoding under the tested configuration.

One-hot encoding avoids imposing equal log-odds effects for successive ordinal codes.

However, the observed differences do not establish statistical or practical superiority in other datasets or future periods.

## Provisional decision

Retain one-hot encoding of Age Group in the initial primary Logistic Regression baseline.

This decision is supported by the feature semantics and Train-only cross-validation results.

It is not a claim that one-hot encoding is universally optimal.

## Limitations

Age Group is an ordinal summary rather than a precise continuous age measure.

The age-code representation is specific to this dataset.

Only one initial regularization configuration was compared.

Fold-level results are not independent observations or formal confidence intervals.

The split is not chronological and lacks verified customer identifiers.

No final Test evaluation or probability calibration has been performed.

## Reproduction

Run from the repository root with the documented environment and verified raw dataset:

    .\.venv\Scripts\python.exe -m experiments.age_encoding_sensitivity

Run automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

## Next steps

- Examine reasonable regularization settings using Train-only group-aware CV.
- Consolidate feature-sensitivity findings.
- Specify a probability-calibration protocol before using Calibration.
- Preserve Test for final evaluation.
