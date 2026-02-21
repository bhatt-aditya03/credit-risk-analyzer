from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model_xgb.pkl")

app = FastAPI(title = "Credit Risk Analyzer API")

class ApplicantData(BaseModel):
    age_years: float
    years_employed: float
    amt_income_total: float
    amt_credit: float
    amt_annuity: float
    cnt_children: int

@app.get("/")
def root():
    return {"message": "Credit Risk Analyzer API is running"}

@app.post("/predict")
def predict_credit_score(data: ApplicantData):

    credit_income_ratio = data.amt_credit / data.amt_income_total
    annuity_income_ratio = data.amt_annuity / data.amt_income_total
    credit_term = data.amt_annuity / data.amt_credit

    features = np.array([[
        data.age_years,
        data.years_employed,
        data.amt_income_total,
        data.amt_credit,
        data.amt_annuity,
        credit_income_ratio,
        annuity_income_ratio,
        credit_term,
        data.cnt_children
    ]])

    risk_probability = model.predict_proba(features)[0][1]

    credit_score = int(300 + (1 - risk_probability) * 600)

    if risk_probability > 0.5:
        decision = "REJECTED"
    elif risk_probability > 0.3:
        decision = "MANUAL REVIEW"
    else:
        decision = "APPROVED"

    return {
        "credit_score": credit_score,
        "risk_probability": round(float(risk_probability), 3),
        "decision": decision
    }
