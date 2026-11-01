# Split Protocol — Version 1

Status: Frozen candidate, pending Git review.

## Objective

Create reproducible Train, Calibration, and Test partitions
for churn prediction while keeping identical predictor patterns
within the same partition.

## Dataset

UCI Iranian Churn Dataset (ID 563)
Total rows: 3150

Raw CSV SHA-256:
395d04854ce6daa1cdee8ab820df5a90691de9277dbba72096b8b01a60a655d1

The raw file is not tracked in Git.

## Group definition

Groups are defined by identical values across the ten
approved primary predictor columns.

The following columns are excluded from group construction:
- Churn (target)
- Status (excluded from primary model)
- Customer Value (decision-layer variable)
- Age (redundant with Age Group)

Observed group diagnostics:
- Distinct patterns: 2826
- Repeated patterns: 168
- Rows in repeated patterns: 492
- Largest group: 12 rows

These groups are predictor-pattern groups, NOT verified
customer identities.

## Splitting method

Method: StratifiedGroupKFold
Number of folds: 5
Shuffle: True
Random state: 42

Fold allocation:
- Train: folds 0, 1, 2
- Calibration: fold 3
- Test: fold 4

## Final partition summary

Train: 1894 rows, 299 churn, 15.79% churn
Calibration: 628 rows, 98 churn, 15.61% churn
Test: 628 rows, 98 churn, 15.61% churn

No predictor-pattern group appears in multiple folds.
All rows have exactly one assignment.

## Stored artifacts

data/splits/v1_manifest.csv
data/splits/v1_metadata.json

The manifest uses zero-based row positions, so the source
CSV's byte-level identity and row order matter.

## Reproduction

Run from the project root with the required dependencies installed:

python experiments/create_split_manifest.py
python experiments/validate_split_manifest.py

Validation checks:
- Source data SHA-256
- Row assignment integrity
- Fold-to-split mapping
- Partition sizes
- Predictor-pattern group isolation

## Model development rules

- Fit preprocessing only on the training portion.
- Use group-aware internal validation within Train for model selection.
- Use Calibration for probability calibration, following a
  pre-specified calibration procedure.
- Do not use Test to select models, thresholds, or business policies.
- Use Test for final evaluation after development decisions are frozen.

## Limitations

This is not a chronological holdout.
No verified customer identifier is available.
Group separation does not prove customer-level independence.

Initial exploratory data audits examined the full dataset.
Future model selection must not use the held-out Test labels.

No retention treatment outcomes are available.
Simulated business utility is not observed causal impact.

## References

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

https://scikit-learn.org/stable/common_pitfalls.html

https://doi.org/10.24432/C5JW3Z
