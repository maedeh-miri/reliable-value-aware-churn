"""Reusable validation of a frozen group-aware split."""

import hashlib
import json
from pathlib import Path

import pandas as pd


def validate_split(
    data_path,
    manifest_path,
    metadata_path,
    expected_counts=None,
):
    """Validate dataset identity, assignments, and group isolation."""

    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    metadata_path = Path(metadata_path)

    # 1. Load the input files.
    df = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    # 2. Verify the original dataset identity.
    actual_hash = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()

    if actual_hash != metadata["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 mismatch.")

    # 3. Verify required manifest columns.
    required_columns = {"row_index", "fold", "split"}

    if not required_columns.issubset(manifest.columns):
        raise ValueError("Missing required manifest columns.")

    # 4. Verify complete row assignments.
    expected_rows = metadata["row_count"]

    if len(df) != expected_rows:
        raise ValueError("Dataset row count mismatch.")

    if len(manifest) != expected_rows:
        raise ValueError("Manifest row count mismatch.")

    expected_indices = list(range(expected_rows))

    if manifest["row_index"].tolist() != expected_indices:
        raise ValueError("Invalid or incomplete row assignments.")

    # 5. Verify the predefined fold-to-split mapping.
    fold_to_split = {
        int(fold): split
        for fold, split in metadata["fold_to_split"].items()
    }

    expected_labels = manifest["fold"].map(fold_to_split)

    if expected_labels.isna().any():
        raise ValueError("Invalid fold assignment.")

    if not expected_labels.equals(manifest["split"]):
        raise ValueError("Invalid split labels or fold mapping.")

    # 6. Optionally verify expected split sizes.
    if expected_counts is not None:
        actual_counts = manifest["split"].value_counts().to_dict()

        if actual_counts != expected_counts:
            raise ValueError("Unexpected split sizes.")

    # 7. Verify the grouping feature contract.
    excluded_columns = [
        "Churn",
        "Status",
        "Customer Value",
        "Age",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    if feature_columns != metadata["grouping_columns"]:
        raise ValueError("Grouping feature columns changed.")

    # 8. Reconstruct groups independently.
    groups = df.groupby(
        feature_columns,
        dropna=False,
    ).ngroup()

    group_check = pd.DataFrame({
        "group": groups,
        "fold": manifest["fold"],
        "split": manifest["split"],
    })

    fold_counts = group_check.groupby("group")["fold"].nunique()
    split_counts = group_check.groupby("group")["split"].nunique()

    if not (fold_counts == 1).all():
        raise ValueError("A group appears in multiple folds.")

    if not (split_counts == 1).all():
        raise ValueError("A group appears in multiple splits.")

    return {
        "row_count": expected_rows,
        "split_counts": manifest["split"].value_counts().to_dict(),
        "group_count": groups.nunique(),
    }
