# Initial Logistic Regression Baseline

## Objective

Evaluate whether a regularized Logistic Regression model can improve churn prediction over the prior-based DummyClassifier.

This is an initial learning-based baseline, not the selected final model.

## Data boundary

Only the frozen v1 Train partition is used for model fitting and internal cross-validation.

Train contains 1894 rows, including 299 positive Churn cases.

The Calibration and Test partitions are not used for model fitting, model selection, threshold optimization, or baseline evaluation.

Before evaluation, the script validates the frozen v1 dataset and split contract.

## Predictors

The model uses the ten approved primary predictors defined in the prediction contract.

Seven numeric predictors are standardized:

- Call Failure
- Subscription Length
- Charge Amount
- Seconds of Use
- Frequency of use
- Frequency of SMS
- Distinct Called Numbers

Complains is passed through as a binary predictor.

Age Group and Tariff Plan are one-hot encoded.

The original CSV column names, including their internal spaces, are preserved in code.

Charge Amount is treated as numeric in this initial baseline, although its exact measurement semantics remain uncertain.

## Preprocessing and leakage prevention

A scikit-learn Pipeline combines ColumnTransformer and LogisticRegression.

StandardScaler is fitted only on the fit portion of each internal fold.

OneHotEncoder is fitted only on the fit portion and uses handle_unknown="ignore".

Validation observations are transformed using the preprocessing parameters learned from the corresponding fit portion.

A fresh Pipeline is created for every fold.

## Model configuration

- Estimator: LogisticRegression
- Regularization: L2
- C: 1.0
- Solver: lbfgs
- Maximum iterations: 1000
- Class weighting: None

These settings are initial fixed choices, not optimized hyperparameters.

## Cross-validation protocol

- Method: StratifiedGroupKFold
- Internal folds: 5
- Shuffle: True
- Random state: 42
- Data: frozen v1 Train only
- Group definition: exact equality across the ten approved predictors

Predictor-pattern groups remain isolated between fit and validation portions.

## Mean cross-validation results

| Metric | Dummy | Logistic Regression |
| --- | ---: | ---: |
| Accuracy | 0.8422 | 0.8986 |
| Precision | 0.0000 | 0.8432 |
| Recall | 0.0000 | 0.4414 |
| ROC-AUC | 0.5000 | 0.9288 |
| Average Precision | 0.1578 | 0.7589 |
| Brier Score | 0.1329 | 0.0715 |
| Log Loss | 0.4361 | 0.2294 |

Values are unweighted arithmetic means of the five fold-level metrics, rounded to four decimal places.

## Variation across folds

| Metric | Standard deviation |
| --- | ---: |
| Accuracy | 0.0042 |
| Precision | 0.0493 |
| Recall | 0.0208 |
| ROC-AUC | 0.0056 |
| Average Precision | 0.0207 |
| Brier Score | 0.0027 |
| Log Loss | 0.0058 |

These standard deviations describe variation across the five internal folds. They are not confidence intervals for future performance.

## Interpretation

Logistic Regression outperformed the DummyClassifier on all reported mean metrics.

ROC-AUC and Average Precision indicate substantially better ranking discrimination in internal Train cross-validation.

Brier Score and Log Loss also improved relative to the Dummy baseline.

However, positive-class recall at the default classification rule was only approximately 0.44.

The classification threshold has not been optimized.

Improved probability losses do not establish that the model is sufficiently calibrated for retention decisions.

## Optimization diagnostics

The solver required between 33 and 36 iterations across the five folds, below the configured maximum of 1000.

No convergence warnings were observed during the reported run.

## Limitations

The results come from internal cross-validation within Train, not the final Test partition.

The current split is not chronological and does not use verified customer identifiers.

Group isolation does not prove complete independence between customer records.

Feature availability at the intended prediction time remains a methodological assumption requiring explicit documentation.

The high predictive scores do not rule out unrecognized leakage or dataset-specific artifacts.

No causal retention effects or observed intervention outcomes are available.

No probability calibration, threshold selection, or business-policy optimization has been performed.

## Reproduction

Run from the repository root with the documented environment and verified raw dataset:

    .\.venv\Scripts\python.exe -m experiments.logistic_baseline_cv

Run the automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

## Next steps

- Evaluate whether model configuration changes are justified using Train-only group-aware CV.
- Examine sensitivity to feature encoding and assumptions.
- Develop a predefined probability-calibration protocol using the reserved Calibration partition.
- Preserve Test for evaluation after development decisions are frozen.
