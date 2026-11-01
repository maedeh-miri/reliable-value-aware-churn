import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


DATA_PATH = Path("data/raw/iranian_churn.csv")
OUTPUT_DIR = Path("data/splits")

EXCLUDED_COLUMNS = [
    "Churn",
    "Status",
    "Customer Value",
    "Age",
]

FOLD_TO_SPLIT = {
    0: "train",
    1: "train",
    2: "train",
    3: "calibration",
    4: "test",
}

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=EXCLUDED_COLUMNS)
y = df["Churn"]

groups = X.groupby(
    list(X.columns),
    dropna=False,
).ngroup()

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fold_assignments = pd.Series(
    -1,
    index=df.index,
    dtype=int,
)

for fold_number, (_, holdout_indices) in enumerate(
    splitter.split(X, y, groups)
):
    fold_assignments.iloc[holdout_indices] = fold_number

split_labels = fold_assignments.map(FOLD_TO_SPLIT)

group_split_counts = pd.DataFrame({
    "group": groups,
    "split": split_labels,
}).groupby("group")["split"].nunique()

if not (group_split_counts == 1).all():
    raise ValueError("A group appears in multiple splits.")

if split_labels.isna().any():
    raise ValueError("Some rows have no split assignment.")

manifest = pd.DataFrame({
    "row_index": range(len(df)),
    "fold": fold_assignments,
    "split": split_labels,
})

dataset_sha256 = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

metadata = {
    "dataset_sha256": dataset_sha256,
    "row_count": len(df),
    "random_state": 42,
    "n_splits": 5,
    "sklearn_version": sklearn.__version__,
    "pandas_version": pd.__version__,
    "grouping_columns": list(X.columns),
    "fold_to_split": FOLD_TO_SPLIT,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = OUTPUT_DIR / "v1_manifest.csv"
metadata_path = OUTPUT_DIR / "v1_metadata.json"

csv_content = manifest.to_csv(
    index=False,
    lineterminator="\n",
)

if manifest_path.exists():
    if manifest_path.read_text(encoding="utf-8") != csv_content:
        raise RuntimeError("Existing manifest differs. Refusing to overwrite.")

manifest_path.write_text(csv_content, encoding="utf-8")

metadata_content = json.dumps(metadata, indent=2) + "\n"

if metadata_path.exists():
    existing_metadata = metadata_path.read_text(encoding="utf-8")

    if existing_metadata != metadata_content:
        raise RuntimeError(
            "Existing metadata differs. Refusing to overwrite."
        )

metadata_path.write_text(
    metadata_content,
    encoding="utf-8",
)

print("Manifest saved:", manifest_path)
print("Metadata saved:", metadata_path)
print("Dataset SHA-256:", dataset_sha256)

print("\nSplit counts:")
print(manifest["split"].value_counts())

print("\nGroup overlap check: PASSED")
print("Unassigned rows check: PASSED")
