"""Evaluate Logistic Regression using Train-only group-aware CV."""

import json
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiments.validate_split_manifest import main as validate_frozen_split


DATA_PATH = Path("data/raw/iranian_churn.csv")
MANIFEST_PATH = Path("data/splits/v1_manifest.csv")
METADATA_PATH = Path("data/splits/v1_metadata.json")


NUMERIC_FEATURES = [
    "Call  Failure",
    "Subscription  Length",
    "Charge  Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
]

BINARY_FEATURES = [
    "Complains",
]

CATEGORICAL_FEATURES = [
    "Age Group",
    "Tariff Plan",
]


def build_logistic_pipeline():
    """Create a fresh preprocessing and classification pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_logistic_cv(X, y, groups):
    """Evaluate a fresh logistic pipeline in each group-aware fold."""

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold_number, (fit_indices, validation_indices) in enumerate(
        cv.split(X, y, groups)
    ):
        X_fit = X.iloc[fit_indices]
        y_fit = y.iloc[fit_indices]

        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        fit_groups = set(groups.iloc[fit_indices])
        validation_groups = set(groups.iloc[validation_indices])

        if not fit_groups.isdisjoint(validation_groups):
            raise ValueError("Group leakage detected.")

        model = build_logistic_pipeline()
        model.fit(X_fit, y_fit)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        optimization_iterations = model.named_steps[
            "classifier"
        ].n_iter_[0]

        fold_result = {
            "fold": fold_number,
            "fit_rows": len(fit_indices),
            "validation_rows": len(validation_indices),
            "accuracy": accuracy_score(
                y_validation,
                predictions,
            ),
            "precision": precision_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_validation,
                probabilities,
            ),
            "average_precision": average_precision_score(
                y_validation,
                probabilities,
            ),
            "brier": brier_score_loss(
                y_validation,
                probabilities,
            ),
            "log_loss": log_loss(
                y_validation,
                probabilities,
                labels=[0, 1],
            ),
            "iterations": optimization_iterations,
        }

        results.append(fold_result)

    return pd.DataFrame(results)


def main():
    """Validate v1 and evaluate Logistic Regression within Train."""

    validate_frozen_split()

    data = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    train_indices = manifest.loc[
        manifest["split"] == "train",
        "row_index",
    ]

    train_data = data.iloc[train_indices].reset_index(drop=True)

    feature_columns = metadata["grouping_columns"]

    expected_features = (
        NUMERIC_FEATURES
        + BINARY_FEATURES
        + CATEGORICAL_FEATURES
    )

    if set(feature_columns) != set(expected_features):
        raise ValueError("Feature contract mismatch.")

    X = train_data[feature_columns]
    y = train_data["Churn"]

    groups = X.groupby(
        feature_columns,
        dropna=False,
    ).ngroup()

    results = evaluate_logistic_cv(X, y, groups)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "average_precision",
        "brier",
        "log_loss",
    ]

    print("\nLogistic Regression — Group-Aware CV")
    print(results.round(4).to_string(index=False))

    print("\nMean metrics:")
    print(results[metric_columns].mean().round(4))

    print("\nStandard deviation across folds:")
    print(results[metric_columns].std().round(4))

    print("\nOptimization iterations:")
    print(results[["fold", "iterations"]].to_string(index=False))


if __name__ == "__main__":
    main()
