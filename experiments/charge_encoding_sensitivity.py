"""Compare numeric and one-hot encoding of Charge Amount."""

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
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    build_logistic_pipeline,
)
from experiments.validate_split_manifest import main as validate_frozen_split


DATA_PATH = Path("data/raw/iranian_churn.csv")
MANIFEST_PATH = Path("data/splits/v1_manifest.csv")
METADATA_PATH = Path("data/splits/v1_metadata.json")


def build_onehot_charge_pipeline():
    """Treat Charge Amount as a categorical predictor."""

    numeric_features = [
        column
        for column in NUMERIC_FEATURES
        if column != "Charge  Amount"
    ]

    categorical_features = (
        CATEGORICAL_FEATURES + ["Charge  Amount"]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
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
    """Calculate probability-based classification metrics."""

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


def evaluate_charge_encoding(X, y, groups):
    """Compare both encodings on identical validation folds."""

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
        y_fit = y.iloc[fit_indices]

        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        numeric_model = build_logistic_pipeline()
        onehot_model = build_onehot_charge_pipeline()

        numeric_model.fit(X_fit, y_fit)
        onehot_model.fit(X_fit, y_fit)

        numeric_probabilities = numeric_model.predict_proba(
            X_validation
        )[:, 1]

        onehot_probabilities = onehot_model.predict_proba(
            X_validation
        )[:, 1]

        numeric_metrics = calculate_metrics(
            y_validation,
            numeric_probabilities,
        )

        onehot_metrics = calculate_metrics(
            y_validation,
            onehot_probabilities,
        )

        fold_result = {"fold": fold_number}

        for metric_name in numeric_metrics:
            numeric_value = numeric_metrics[metric_name]
            onehot_value = onehot_metrics[metric_name]

            fold_result[f"numeric_{metric_name}"] = numeric_value
            fold_result[f"onehot_{metric_name}"] = onehot_value
            fold_result[f"delta_{metric_name}"] = (
                onehot_value - numeric_value
            )

        results.append(fold_result)

    return pd.DataFrame(results)


def main():
    """Evaluate Charge Amount encoding within frozen Train."""

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

    X = train_data[feature_columns]
    y = train_data["Churn"]

    groups = X.groupby(
        feature_columns,
        dropna=False,
    ).ngroup()

    results = evaluate_charge_encoding(X, y, groups)

    print("\nCharge Amount Encoding Sensitivity")

    for metric_name in [
        "roc_auc",
        "average_precision",
        "brier",
        "log_loss",
    ]:
        columns = [
            "fold",
            f"numeric_{metric_name}",
            f"onehot_{metric_name}",
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
