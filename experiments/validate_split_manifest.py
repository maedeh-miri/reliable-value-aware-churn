import hashlib
import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/iranian_churn.csv")
MANIFEST_PATH = Path("data/splits/v1_manifest.csv")
METADATA_PATH = Path("data/splits/v1_metadata.json")

df = pd.read_csv(DATA_PATH)
manifest = pd.read_csv(MANIFEST_PATH)

metadata = json.loads(
    METADATA_PATH.read_text(encoding="utf-8")
)

# 1. Verify the raw dataset
actual_hash = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

assert actual_hash == metadata["dataset_sha256"]

print("Dataset SHA-256: PASSED")


# 2. Verify row assignments
assert len(df) == len(manifest) == 3150
assert manifest["row_index"].tolist() == list(range(len(df)))

print("Row assignment integrity: PASSED")


# 3. Verify the predefined fold mapping
fold_to_split = {
    0: "train",
    1: "train",
    2: "train",
    3: "calibration",
    4: "test",
}

assert manifest["fold"].map(fold_to_split).equals(
    manifest["split"]
)

print("Fold mapping: PASSED")


# 4. Verify expected split sizes
expected_counts = {
    "train": 1894,
    "calibration": 628,
    "test": 628,
}

actual_counts = manifest["split"].value_counts().to_dict()

assert actual_counts == expected_counts

print("Split sizes: PASSED")


# 5. Reconstruct groups independently
excluded_columns = [
    "Churn",
    "Status",
    "Customer Value",
    "Age",
]

feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

assert feature_columns == metadata["grouping_columns"]

groups = df.groupby(
    feature_columns,
    dropna=False,
).ngroup()

group_check = pd.DataFrame({
    "group": groups,
    "fold": manifest["fold"],
    "split": manifest["split"],
})

fold_overlap = group_check.groupby("group")["fold"].nunique()
split_overlap = group_check.groupby("group")["split"].nunique()

assert (fold_overlap == 1).all()
assert (split_overlap == 1).all()

print("Group isolation: PASSED")

print("\nAll split validation checks passed.")
