import pandas as pd


DATA_PATH = "data/raw/iranian_churn.csv"


df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique rows after exact deduplication:")
print(len(df.drop_duplicates()))

duplicate_mask = df.duplicated(keep=False)

duplicate_rows = df[duplicate_mask]

print("\nRows involved in duplicate groups:")
print(len(duplicate_rows))

print("\nChurn distribution among duplicate-group rows:")
print(duplicate_rows["Churn"].value_counts().sort_index())


feature_columns = df.columns.drop("Churn")

feature_duplicate_mask = df.duplicated(
    subset=feature_columns,
    keep=False,
)

feature_duplicate_rows = df[feature_duplicate_mask]

print("\nRows with duplicated feature patterns:")
print(len(feature_duplicate_rows))
print("\nChecking conflicting churn labels:")

churn_options = df.groupby(list(feature_columns))["Churn"].nunique()

conflicting_patterns = churn_options[churn_options > 1]

print("Number of conflicting feature patterns:")
print(len(conflicting_patterns))
pattern_summary = df.groupby(list(feature_columns))["Churn"].agg(
    row_count="size",
    distinct_labels="nunique",
)

conflicting_group_summary = pattern_summary[
    pattern_summary["distinct_labels"] > 1
]

print("\nTotal rows belonging to conflicting feature patterns:")
print(conflicting_group_summary["row_count"].sum())
print("\nDistribution of conflicting group sizes:")
print(
    conflicting_group_summary["row_count"]
    .value_counts()
    .sort_index()
)
label_types_per_row = df.groupby(
    list(feature_columns)
)["Churn"].transform("nunique")

conflicting_rows = df[label_types_per_row > 1]

print("\nChurn distribution within conflicting patterns:")
print(conflicting_rows["Churn"].value_counts().sort_index())

print("\nVerification - total conflicting rows:")
print(len(conflicting_rows))
