import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DATA_PATH = "data/raw/iranian_churn.csv"

EXCLUDED_COLUMNS = [
    "Churn",
    "Status",
    "Customer Value",
    "Age",
]

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

fold_assignments = pd.Series(-1, index=df.index)

for fold_number, (_, holdout_indices) in enumerate(
    splitter.split(X, y, groups)
):
    fold_assignments.iloc[holdout_indices] = fold_number

print("Five-fold split diagnostics:")
print("Overall churn rate:", round(y.mean() * 100, 2), "%")

for fold_number in range(5):
    mask = fold_assignments == fold_number

    print(
        f"Fold {fold_number}: "
        f"rows={mask.sum()}, "
        f"churn={y[mask].sum()}, "
        f"churn_rate={y[mask].mean() * 100:.2f}%"
    )

group_fold_counts = pd.DataFrame({
    "group": groups,
    "fold": fold_assignments,
}).groupby("group")["fold"].nunique()

print("\nGroups appearing in multiple folds:")
print((group_fold_counts > 1).sum())

print("\nUnassigned rows:")
print((fold_assignments == -1).sum())
fold_to_split = {
    0: "train",
    1: "train",
    2: "train",
    3: "calibration",
    4: "test",
}

split_labels = fold_assignments.map(fold_to_split)

print("\nProposed Train / Calibration / Test split:")

for split_name in ["train", "calibration", "test"]:
    mask = split_labels == split_name

    print(
        f"{split_name}: "
        f"rows={mask.sum()}, "
        f"churn={y[mask].sum()}, "
        f"churn_rate={y[mask].mean() * 100:.2f}%"
    )

group_split_counts = pd.DataFrame({
    "group": groups,
    "split": split_labels,
}).groupby("group")["split"].nunique()

print("\nGroups appearing in multiple final splits:")
print((group_split_counts > 1).sum())

print("\nRows without a split:")
print(split_labels.isna().sum())

assert (group_split_counts == 1).all()
assert split_labels.notna().all()
