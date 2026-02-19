import pandas as pd

print("Loading Dataset...")

df = pd.read_csv("application_train.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows.")
print(df.head())

print("\n Target column distribution.")
print(df["TARGET"].value_counts())

print("\n Missing values in first 10 columns.")
print(df.iloc[:, :10].isnull().sum())