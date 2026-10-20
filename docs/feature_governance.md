# Feature Governance

Status: Draft v0.1

## Source

UCI Iranian Churn Dataset, ID 563
https://doi.org/10.24432/C5JW3Z

## Primary prediction features

Candidate primary model predictors (10):

- Call Failure
- Complains
- Subscription Length
- Charge Amount
- Seconds of Use
- Frequency of use
- Frequency of SMS
- Distinct Called Numbers
- Age Group
- Tariff Plan

These are the human-readable column names.
Some raw CSV headers contain additional spaces.

## Feature types

Count or duration:
Call Failure, Subscription Length, Seconds of Use,
Frequency of use, Frequency of SMS, Distinct Called Numbers.

Binary:
Complains, Tariff Plan.

Ordinal:
Charge Amount, Age Group.

Encoding and scaling choices are not yet finalized.

## Age redundancy

The observed dataset has a one-to-one mapping:

Age Group 1 -> Age 15
Age Group 2 -> Age 25
Age Group 3 -> Age 30
Age Group 4 -> Age 45
Age Group 5 -> Age 55

Retain Age Group; exclude Age from the primary model.

This is an observed property of this dataset, not a
general statement about customer ages.

## Charge Amount documentation discrepancy

UCI documentation describes ordinal levels 0 through 9.

The local dataset contains levels 0 through 10.
Exactly 7 rows have Charge Amount equal to 10.

No values have been modified or removed.
The discrepancy requires explicit documentation.

## Other feature exclusions

Status:
Excluded from the conservative primary model because
its semantics are close to customer inactivity.
A separate sensitivity analysis is planned.
It has not been established as definite leakage.

Customer Value:
Excluded from primary churn predictors.
Reserved for a separate decision-prioritization layer.
Its monetary unit and economic interpretation are unverified.

Churn:
Prediction target; never a predictor.

## Data governance notes

Raw data remains unchanged.

Feature auditing was exploratory and used the full
dataset before creating the final evaluation split.
No model-selection decisions should be based on
full-dataset target associations.

These choices will be reviewed before model training.
