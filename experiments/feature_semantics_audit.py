import pandas as pd

df = pd.read_csv("data/raw/iranian_churn.csv")

columns_to_check = [
    "Complains",
    "Tariff Plan",
    "Status",
    "Age Group",
]

for column in columns_to_check:
    print(f"\nValues of {column}:")
    print(df[column].value_counts().sort_index())

print("\nStatus versus Churn:")
print(pd.crosstab(df["Status"], df["Churn"]))

print("\nAge and Customer Value summary:")
print(df[["Age", "Customer Value"]].describe().round(2))
