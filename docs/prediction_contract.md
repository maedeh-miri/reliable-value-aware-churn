# Prediction Contract

Status: Draft v0.1 — subject to validation before model training.

## 1. Research question

Can calibrated churn probabilities support reliable,
value-aware customer retention decisions under limited budgets?

## 2. Prediction target

Target column: Churn
Positive class: 1 (churn)
Negative class: 0 (non-churn)

## 3. Prediction horizon

The dataset describes customer behavior during the first nine months.
Churn is recorded at the end of month twelve.

Intended prediction point: end of month 9.
Intended outcome horizon: end of month 12.

Source: https://doi.org/10.24432/C5JW3Z

## 4. Primary feature policy

Primary churn model:
- Exclude Status as a conservative feature-governance choice.
- Exclude Customer Value from the primary predictor set.
- Exclude Age because it duplicates Age Group in the observed dataset.
- Audit the remaining 10 predictors before freezing preprocessing.

Status will be examined separately in a sensitivity analysis.

Customer Value will initially be treated as a relative
customer-priority score, not verified monetary profit or CLV.

## 5. Data integrity

The initial audit found:
- 3,150 rows and 14 columns
- 300 extra exact duplicate copies
- 14 conflicting feature patterns involving 64 rows

No rows have been removed automatically.

## 6. Evaluation design

Proposed Train / Calibration / Test allocation: 60% / 20% / 20%.

Before creating the split:
- Evaluate grouping based on repeated predictor patterns.
- Check class balance and group overlap.
- Record and verify the final split.
- Keep the final test set out of model selection.

This split policy is provisional.

## 7. Interpretation boundaries

The dataset lacks observed retention-intervention outcomes.
Decision utility will be simulated under explicit assumptions.

The project will not claim causal treatment effects
or verified realized retention profit.
