# Dummy Churn Baseline

## Objective

Establish a simple, reproducible reference for future churn prediction models.

The baseline uses DummyClassifier with strategy="prior". It ignores customer predictors and estimates class probabilities from the class frequencies in each training fold.

## Data boundary

Only the frozen v1 Train partition is used for model fitting and internal cross-validation.

Train contains 1894 rows, including 299 positive Churn cases.

The Calibration and Test partitions are not used for model fitting, model selection, or baseline evaluation.

Before running the baseline, the script validates the frozen v1 dataset and split contract.

## Evaluation protocol

- Method: StratifiedGroupKFold
- Internal folds: 5
- Shuffle: True
- Random state: 42
- Group definition: exact equality across the ten approved predictors
- Fit: four internal folds
- Validation: one held-out internal fold

Each internal validation row is evaluated once.

Predictor-pattern groups remain isolated between the fit and validation portions.

## Mean cross-validation results

| Metric | Mean |
| --- | ---: |
| Accuracy | 0.8422 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| ROC-AUC | 0.5000 |
| Average Precision | 0.1578 |
| Brier Score | 0.1329 |
| Log Loss | 0.4361 |

These are unweighted arithmetic means of the five fold-level metrics, rounded to four decimal places.

## Interpretation

The high accuracy reflects the majority-class prevalence and does not indicate useful detection of churn-positive customers.

The baseline predicts the majority class for every validation observation, resulting in zero positive-class recall.

Constant probabilities within each fold provide no within-fold ranking discrimination, producing ROC-AUC of 0.5.

Average Precision is approximately equal to the positive-class prevalence in each validation fold.

Brier Score and Log Loss establish reference probability-loss values for later models.

## Limitations

These results are development-stage estimates from internal Train cross-validation, not final Test performance.

The baseline has no customer-specific ranking ability.

A low probability loss does not, by itself, prove that the probabilities are sufficiently calibrated for retention decisions.

No threshold optimization, calibration fitting, or business-policy selection has been performed.

The original dataset lacks verified customer identifiers and a chronological holdout.

## Reproduction

Run from the repository root after preparing the documented Python environment and acquiring the verified raw dataset:

    .\.venv\Scripts\python.exe -m experiments.dummy_baseline_cv

Run automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

## Next step

Compare this reference against a leakage-aware Logistic Regression pipeline using group-aware cross-validation within Train.
