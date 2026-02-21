import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

print("Loading data...")
df = pd.read_csv("application_train.csv")

missing = df.isnull().sum()
missing_percent = (missing / len(df) * 100)
cols_to_drop = missing_percent[missing_percent > 50].index.tolist()
df = df.drop(columns = cols_to_drop)

df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365)
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, float("NaN"))
df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365)
df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

features = ["AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL", 
            "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO", 
            "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"]

X = df[features]
y = df["TARGET"]

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter = 1000)
lr_model.fit(X_train, y_train)

y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")

joblib.dump(lr_model, "model_lr.pkl")
print("\nModel saved as model_lr.pkl")


from xgboost import XGBClassifier

print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10, 
    random_state=42,
    eval_metric="auc"
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, y_pred_xgb)
print(f"XGBoost ROC-AUC Score: {xgb_auc:.4f}")

joblib.dump(xgb_model, "model_xgb.pkl")
print("XGBoost model saved as model_xgb.pkl")


import numpy as np

sample = X_test.iloc[:5]
probs = xgb_model.predict_proba(sample)[:, 1]
scores = (300 + (1 - probs) * 600).astype(int)

print("\n=== Sample Credit Scores ===")
for i, (prob, score) in enumerate(zip(probs, scores)):
    status = "REJECTED" if prob > 0.5 else "APPROVED"
    print(f"Applicant {i+1}: Risk={prob:.3f} | Score={score} | {status}")