import pandas as pd

data = {
    "Name":["Aditya", "Rahul", "Priya", "Soniya", "Suresh"],
    "Age":[22, 35, 28, 45, 31],
    "Monthly_income":[25000, 60000, 35000, 80000, 20000],
    "Loan_amount":[200000, 150000, 500000, 100000, 300000],
    "Loan_repaid":[0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("=== First look at data ===")
print(df)

print("\n=== Shape(rows, columns) ===")
print(df.shape)

print("\n=== Basic statistics ===")
print(df.describe())

print("\n=== How many defaulted vs repaid ===")
print(df["Loan_repaid"].value_counts())

df["Annual_income"] = df["Monthly_income"] * 12
df["Debt_to_income_ratio"] = df["Loan_amount"] / df["Annual_income"]
df["Loan_to_income_percent"] = (df["Loan_amount"] / df["Annual_income"]) * 100

print("\n=== After Feature Engineering ===")
print(df[["Name", "Debt_to_income_ratio", "Loan_to_income_percent", "Loan_repaid"]])

print("\n=== High Risk Applicants (ratio > 0.4) ===")
High_risk = df[df["Debt_to_income_ratio"] > 0.4]
print(High_risk[["Name", "Debt_to_income_ratio", "Loan_repaid"]])


