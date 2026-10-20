import pandas as pd


DATA_PATH = "data/raw/iranian_churn.csv"

EXCLUDED_COLUMNS = [
    "Churn",
    "Status",
    "Customer Value",
]

df = pd.read_csv(DATA_PATH)

predictors = df.drop(columns=EXCLUDED_COLUMNS)

print("Number of candidate predictors:")
print(len(predictors.columns))

print("\nCandidate predictor names:")
for column in predictors.columns:
    print(column)

print("\nPredictor summary:")
summary = pd.DataFrame({
    "data_type": predictors.dtypes.astype(str),
    "unique_values": predictors.nunique(),
    "missing_values": predictors.isna().sum(),
    "minimum": predictors.min(),
    "maximum": predictors.max(),
})

print(summary.to_string())

print("\nAge Group versus Age:")
print(
    df.groupby("Age Group")["Age"].agg(
        ["min", "max", "nunique"]
    )
)
print("\nCharge Amount value counts:")
print(df["Charge  Amount"].value_counts().sort_index())

print("\nAge Group versus Age cross-tabulation:")
print(pd.crosstab(df["Age Group"], df["Age"]))
