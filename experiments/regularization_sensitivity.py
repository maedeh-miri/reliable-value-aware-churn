"""Examine Logistic Regression sensitivity to regularization."""

import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from experiments.logistic_baseline_cv import build_logistic_pipeline
from experiments.validate_split_manifest import main as validate_frozen_split


DATA_PATH = Path("data/raw/iranian_churn.csv")
MANIFEST_PATH = Path("data/splits/v1_manifest.csv")
METADATA_PATH = Path("data/splits/v1_metadata.json")

C_VALUES = [0.1, 1.0, 10.0]


def evaluate_regularization(X, y, groups):
    """Evaluate fixed regularization settings on internal Train folds."""

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

        for c_value in C_VALUES:
            model = build_logistic_pipeline()

            model.set_params(
                classifier__C=c_value
            )

            model.fit(X_fit, y_fit)

            probabilities = model.predict_proba(
                X_validation
            )[:, 1]

            iterations = model.named_steps[
                "classifier"
            ].n_iter_[0]

            fold_result = {
                "fold": fold_number,
                "C": c_value,
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
                "iterations": iterations,
            }

            results.append(fold_result)

    return pd.DataFrame(results)


def main():
    """Run regularization sensitivity within frozen Train."""

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

    results = evaluate_regularization(X, y, groups)

    metric_columns = [
        "roc_auc",
        "average_precision",
        "brier",
        "log_loss",
    ]

    print("\nRegularization Sensitivity — Fold Results")
    print(results.round(4).to_string(index=False))

    mean_results = results.groupby("C")[metric_columns].mean()

    print("\nMean metrics by C:")
    print(mean_results.round(4))

    std_results = results.groupby("C")[metric_columns].std()

    print("\nStandard deviation by C:")
    print(std_results.round(4))

    print("\nOptimization iterations:")
    print(
        results[["fold", "C", "iterations"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
