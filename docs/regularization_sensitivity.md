# Regularization Sensitivity Analysis

## Objective

Evaluate whether the initial Logistic Regression results are sensitive to reasonable changes in regularization strength.

This is a Train-only robustness analysis, not a final held-out evaluation.

## Data boundary

All model comparisons use the frozen Train partition:

- Train rows: 1894
- Calibration: reserved and not used
- Test: reserved and not used

The frozen split is validated before the experiment.

## Regularization concept

Regularization discourages excessively large model coefficients.

In scikit-learn LogisticRegression, C is the inverse of regularization strength:

- C = 0.1: stronger regularization
- C = 1.0: initial default setting
- C = 10.0: weaker regularization

These three values form a small, predefined sensitivity range.

They are not claimed to be theoretically optimal.

## Experimental design

The experiment compares three C values:

- 0.1
- 1.0
- 10.0

All other modeling choices remain unchanged.

Each configuration uses the same five group-aware cross-validation folds within Train.

Preprocessing is fitted separately within each training fold to avoid validation leakage.

## Mean cross-validation results

| C | ROC-AUC | Average Precision | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
| 0.1 | 0.9232 | 0.7361 | 0.0763 | 0.2473 |
| 1.0 | 0.9288 | 0.7589 | 0.0715 | 0.2294 |
| 10.0 | 0.9294 | 0.7622 | 0.0711 | 0.2283 |

Higher ROC-AUC and Average Precision are better.

Lower Brier Score and Log Loss are better.

## Variation across folds

Standard deviations across the five folds:

| C | ROC-AUC SD | AP SD | Brier SD | Log Loss SD |
|---|---:|---:|---:|---:|
| 0.1 | 0.0078 | 0.0220 | 0.0025 | 0.0068 |
| 1.0 | 0.0056 | 0.0207 | 0.0027 | 0.0058 |
| 10.0 | 0.0049 | 0.0165 | 0.0032 | 0.0056 |

These values describe variation across the chosen folds.

They are not independent confidence intervals or proof of statistical significance.

## Interpretation

C = 0.1 produces consistently weaker average results than the initial C = 1.0 configuration.

C = 10.0 produces slightly better mean results than C = 1.0.

However, the differences between C = 1.0 and C = 10.0 are small:

- ROC-AUC improvement: 0.0006
- Average Precision improvement: 0.0033
- Brier Score reduction: 0.0004
- Log Loss reduction: 0.0011

The evidence does not establish a practically important advantage for C = 10.0.

## Optimization diagnostics

Observed optimization iterations range from 19 to 55 across the evaluated configurations.

These runs did not show evidence of reaching the configured maximum iteration limit.

## Provisional decision

Retain C = 1.0 as the current reference configuration.

Reasons:

- It is the initial documented setting.
- Its performance is close to C = 10.0.
- Retaining it avoids an additional configuration change based on very small observed differences.

This decision is provisional.

It does not establish that C = 1.0 is globally optimal.

## Limitations

The analysis uses one frozen Train partition and one predefined set of internal cross-validation folds.

Repeated comparisons on the same folds can influence subsequent development decisions.

Fold-level standard deviations do not establish statistical significance.

Probability calibration has not yet been assessed on the reserved Calibration partition.

No Test results are reported or used for configuration selection.

## Reproduction

Run from the repository root:

    .\.venv\Scripts\python.exe -m experiments.regularization_sensitivity

Run automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

## References

scikit-learn LogisticRegression documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

scikit-learn cross-validation documentation:
https://scikit-learn.org/stable/modules/cross_validation.html

## Next steps

- Consolidate the Train-only sensitivity findings.
- Review the provisional model configuration.
- Define the probability-calibration protocol before accessing Calibration.
- Preserve Test for the final evaluation.
