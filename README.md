# Reliable Value-Aware Churn Decisions

A reproducible machine learning portfolio project investigating how
churn predictions can support reliable customer retention decisions.

## Research focus

- Data integrity and feature governance
- Leakage-aware model evaluation
- Probability calibration
- Value-aware retention targeting under limited budgets
- Decision stability and sensitivity analysis

## Dataset

UCI Iranian Churn Dataset, ID 563.
DOI: https://doi.org/10.24432/C5JW3Z

## Current status

Data audits, feature governance, and the frozen group-aware v1 split are complete. Reproducible environment documentation and automated validation tests have been added. A Train-only DummyClassifier baseline has been evaluated with group-aware cross-validation. Logistic Regression and probability calibration remain pending. See [Dummy Baseline](docs/dummy_baseline.md).

## Reproducing the initial audits

Run these commands from the repository root after installing pandas:

python experiments/data_integrity_audit.py
python experiments/feature_semantics_audit.py

## Important limitations

The dataset does not provide observed retention-intervention outcomes.
Any future intervention utility calculations will therefore rely on
explicit simulation assumptions, not causal effect estimates.

## Environment and validation

Reference environment: Windows, Python 3.13.2.

Direct dependencies are listed in `requirements.txt`.
The recorded environment snapshot is in `requirements-lock.txt`.

For setup and reproducibility limitations, see [Environment](docs/environment.md).

### Install dependencies

From the repository root:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

### Validate the frozen split

The original dataset must be available at `data/raw/iranian_churn.csv`.

    .\.venv\Scripts\python.exe -m experiments.validate_split_manifest

For the split design and limitations, see [Split Protocol](docs/split_protocol.md).

### Run automated tests

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

The tests use temporary synthetic data and do not modify the frozen v1 split.

### Check installed dependencies

    .\.venv\Scripts\python.exe -m pip check

## Next steps

- Establish leakage-aware model baselines.
- Evaluate predictive performance and probability calibration.
- Study value-aware retention decisions under limited budgets.
- Assess decision stability and sensitivity to assumptions.
