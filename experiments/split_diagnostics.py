import pandas as pd


DATA_PATH = "data/raw/iranian_churn.csv"

EXCLUDED_COLUMNS = [
    "Churn",
    "Status",
    "Customer Value",
    "Age",
]

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=EXCLUDED_COLUMNS)

group_sizes = X.groupby(
    list(X.columns),
    dropna=False,
).size()

print("Total rows:")
print(len(df))

print("\nPrimary model features:")
print(len(X.columns))

print("\nDistinct predictor patterns:")
print(len(group_sizes))

print("\nPatterns appearing more than once:")
print((group_sizes > 1).sum())

print("\nRows in repeated predictor patterns:")
print(group_sizes[group_sizes > 1].sum())

print("\nLargest group size:")
print(group_sizes.max())

print("\nTop 10 group sizes:")
print(group_sizes.sort_values(ascending=False).head(10).tolist())
