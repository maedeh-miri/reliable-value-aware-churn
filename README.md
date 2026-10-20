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

Exploratory data and feature-semantics audits are in progress.

## Reproducing the initial audits

Run these commands from the repository root after installing pandas:

python experiments/data_integrity_audit.py
python experiments/feature_semantics_audit.py

## Important limitations

The dataset does not provide observed retention-intervention outcomes.
Any future intervention utility calculations will therefore rely on
explicit simulation assumptions, not causal effect estimates.
