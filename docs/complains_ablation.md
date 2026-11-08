# Complains Feature Ablation

## Objective

Examine how sensitive the initial Logistic Regression baseline is to the inclusion of the Complains predictor.

This is a predictive sensitivity analysis, not a causal analysis.

## Data boundary

Only the frozen v1 Train partition is used.

Calibration and Test remain excluded from model fitting, evaluation, and model-selection decisions.

The frozen v1 manifest and metadata are not modified.

## Comparison design

Two Logistic Regression pipelines are compared:

- Full model: all ten approved predictors
- Reduced model: the same predictors except Complains

Both use the same initial model configuration:

- L2 regularization
- C = 1.0
- Solver = lbfgs
- Maximum iterations = 1000

Preprocessing is fitted independently within each training fold.

## Group-aware cross-validation

Removing Complains changes the definition of identical predictor patterns.

Within Train:

- Unique groups with ten predictors: 1705
- Unique groups without Complains: 1693
- Reduction in unique groups: 12

For a leakage-aware paired comparison, both models use the same five internal folds constructed from equality across the nine reduced predictors.

The method is StratifiedGroupKFold with shuffle=True and random_state=42.

Each Train row appears exactly once in internal validation, and the reduced-feature groups do not overlap between fit and validation.

## Mean results

| Metric | Full model | Without Complains | Delta |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.9313 | 0.8838 | -0.0476 |
| Average Precision | 0.7622 | 0.5742 | -0.1881 |
| Brier Score | 0.0712 | 0.0927 | +0.0216 |
| Log Loss | 0.2297 | 0.2877 | +0.0579 |

Delta is defined as the reduced-model metric minus the full-model metric.

For ROC-AUC and Average Precision, higher is better.

For Brier Score and Log Loss, lower is better.

Values are rounded to four decimal places.

## Interpretation

Removing Complains reduced ROC-AUC and Average Precision in all five internal folds.

Brier Score and Log Loss also became worse in all five folds.

The result indicates that the predictive performance of this Logistic Regression model is sensitive to the presence of Complains.

The reduced model nevertheless retains substantial discrimination, indicating that other predictors also carry predictive information.

## Important methodological distinction

These paired folds differ from the original ten-feature internal cross-validation folds used for the initial Logistic Regression baseline.

Therefore, both the full and reduced models were refitted and reevaluated on the same nine-feature-grouped folds.

The paired comparison should not be calculated by subtracting results from the earlier baseline experiment.

## Temporal availability evidence

The UCI dataset description states that predictors are aggregated over the first nine months, while Churn is observed at the end of month twelve.

Complains is documented as a binary indicator of whether a complaint occurred.

This supports the intended end-of-month-nine prediction contract.

However, individual complaint timestamps and event-level audit records are unavailable, so feature timing cannot be independently verified per customer.

Source: https://archive.ics.uci.edu/dataset/563/iranian%2Bchurn%2Bdataset

## Limitations

Feature ablation does not establish a causal effect of complaints on churn.

The precise timing and semantics of Complains require continued scrutiny against the intended end-of-month-nine prediction point.

The dataset has no verified customer identifiers or chronological holdout.

The observed sensitivity may not generalize to new periods, customers, or operational settings.

The results are from Train-only cross-validation and do not establish final Test performance.

No probability calibration or retention-policy optimization has been performed.

## Reproduction

Run from the repository root with the documented environment and verified raw dataset:

    .\.venv\Scripts\python.exe -m experiments.complains_ablation_cv

Run automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

## Next steps

- Review the feature's temporal availability and source semantics.
- Examine sensitivity to Charge Amount encoding.
- Examine sensitivity to Age Group encoding.
- Preserve Calibration and Test for their predefined later roles.
