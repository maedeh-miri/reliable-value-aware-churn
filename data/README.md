# Dataset Provenance and Initial Audit

## Source

UCI Iranian Churn Dataset (ID 563)
DOI: https://doi.org/10.24432/C5JW3Z
License: CC BY 4.0

The dataset was retrieved using the ucimlrepo Python package.
The local snapshot is stored at data/raw/iranian_churn.csv.
Raw data is excluded from Git tracking.

According to UCI, features describe the first nine months,
while churn is assessed at the end of month twelve.

## Initial audit findings

- Observations: 3,150
- Columns: 14, including target
- Missing cells: 0
- Churn = 0: 2,655
- Churn = 1: 495
- Extra exact duplicate copies: 300
- Unique complete rows: 2,850
- Rows participating in exact duplicate groups: 465
- Rows participating in repeated feature patterns: 476
- Feature patterns associated with both churn labels: 14
- Rows within conflicting feature patterns: 64
- Churn distribution within those 64 rows: 32 zeros and 32 ones

## Open methodological questions

- Repeated rows may represent distinct customers; no automatic
  deduplication has been performed.
- Split design must account for potentially overlapping feature patterns.
- Status is excluded from the proposed conservative primary predictor
  set pending a documented feature-governance decision.
- Customer Value has no documented currency unit and should not be
  described as monetary profit or verified lifetime value.
- These results are exploratory, calculated before train/test separation.
