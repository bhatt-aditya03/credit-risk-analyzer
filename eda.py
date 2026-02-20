import pandas as pd

print("Loading dataset...")
df = pd.read_csv("application_train.csv")

print("=== Missing Values (columns with missing data only) ===")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_percent = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})
print(missing_report.head(20))

print("\n=== Column Data Types ===")
print(df.dtypes.value_counts())

key_cols = ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", 
            "AMT_ANNUITY", "AGE_DAYS", "DAYS_EMPLOYED"]

key_cols = ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT",
            "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]

print("\n=== Key Columns Statistics ===")
print(df[key_cols].describe())

df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(int)
print("\n=== Age Distribution ===")
print(df["AGE_YEARS"].describe())

df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, float("nan"))
df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).round(1)
print("\n=== Years Employed (after fix) ===")
print(df["YEARS_EMPLOYED"].describe())

cols_to_drop = missing[missing_percent > 50].index.tolist()
print(f"\n=== Dropping {len(cols_to_drop)} columns with >50% missing ===")
df = df.drop(columns=cols_to_drop)
print(f"New shape: {df.shape}")

df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

print("\n=== New Features vs Default Rate ===")
print(df.groupby("TARGET")[["CREDIT_INCOME_RATIO", 
                             "ANNUITY_INCOME_RATIO", 
                             "AGE_YEARS"]].mean().round(3))