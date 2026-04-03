import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

print("Loading data...")
df = pd.read_csv("application_train.csv")
print(f"Training on {df.shape[0]} samples, {df.shape[1]} features")

# Drop columns with >50% missing values
threshold = 0.5
df = df[df.columns[df.isnull().mean() < threshold]]

# Fix DAYS_EMPLOYED sentinel value (365243 = unemployed/pensioner)
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

# Feature engineering
df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365
df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

# Select features
FEATURES = [
    "AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"
]

X = df[FEATURES]
y = df["TARGET"]

# Train/test split BEFORE imputation to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit medians on training data only
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

# Logistic Regression baseline
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
joblib.dump(lr_model, "model_lr.pkl")

# XGBoost
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
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
joblib.dump(xgb_model, "model_xgb.pkl")
print("\nModels saved.")