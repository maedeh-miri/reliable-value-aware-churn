"""Compare logistic churn models with and without Complains."""

import json
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiments.logistic_baseline_cv import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_logistic_pipeline,
)
from experiments.validate_split_manifest import main as validate_frozen_split


DATA_PATH = Path("data/raw/iranian_churn.csv")
MANIFEST_PATH = Path("data/splits/v1_manifest.csv")
METADATA_PATH = Path("data/splits/v1_metadata.json")


def build_reduced_pipeline():
    """Build Logistic Regression without the Complains feature."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
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


def calculate_metrics(y_true, probabilities):
    """Calculate probability-based evaluation metrics."""

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
    }


def evaluate_ablation(X, y, groups):
    """Compare both models on identical validation folds."""

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold_number, (fit_indices, validation_indices) in enumerate(
        cv.split(X, y, groups)
    ):
        fit_groups = set(groups.iloc[fit_indices])
        validation_groups = set(groups.iloc[validation_indices])

        if not fit_groups.isdisjoint(validation_groups):
            raise ValueError("Group leakage detected.")

        X_fit = X.iloc[fit_indices]
        X_validation = X.iloc[validation_indices]

        y_fit = y.iloc[fit_indices]
        y_validation = y.iloc[validation_indices]

        full_model = build_logistic_pipeline()
        reduced_model = build_reduced_pipeline()

        full_model.fit(X_fit, y_fit)

        reduced_model.fit(
            X_fit.drop(columns=["Complains"]),
            y_fit,
        )

        full_probabilities = full_model.predict_proba(
            X_validation
        )[:, 1]

        reduced_probabilities = reduced_model.predict_proba(
            X_validation.drop(columns=["Complains"])
        )[:, 1]

        full_metrics = calculate_metrics(
            y_validation,
            full_probabilities,
        )

        reduced_metrics = calculate_metrics(
            y_validation,
            reduced_probabilities,
        )

        fold_result = {"fold": fold_number}

        for metric_name in full_metrics:
            full_value = full_metrics[metric_name]
            reduced_value = reduced_metrics[metric_name]

            fold_result[f"full_{metric_name}"] = full_value
            fold_result[f"without_{metric_name}"] = reduced_value
            fold_result[f"delta_{metric_name}"] = (
                reduced_value - full_value
            )

        results.append(fold_result)

    return pd.DataFrame(results)


def main():
    """Run the paired ablation within the frozen Train partition."""

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

    full_features = metadata["grouping_columns"]

    reduced_features = [
        column
        for column in full_features
        if column != "Complains"
    ]

    X = train_data[full_features]
    y = train_data["Churn"]

    groups = X.groupby(
        reduced_features,
        dropna=False,
    ).ngroup()

    results = evaluate_ablation(X, y, groups)

    print("\nComplains Ablation — Paired Group-Aware CV")

    for metric_name in [
        "roc_auc",
        "average_precision",
        "brier",
        "log_loss",
    ]:
        columns = [
            "fold",
            f"full_{metric_name}",
            f"without_{metric_name}",
            f"delta_{metric_name}",
        ]

        print(f"\n{metric_name}:")
        print(results[columns].round(4).to_string(index=False))

        print(
            "Mean delta:",
            round(results[f"delta_{metric_name}"].mean(), 4),
        )


if __name__ == "__main__":
    main()
