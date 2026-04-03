from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from features import build_features, compute_score, get_decision

# Load model relative to this file — works from any directory
MODEL_PATH = Path(__file__).parent / "model_xgb.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Credit Risk Analyzer API",
    description="Predicts loan default risk and generates credit scores (300-900).",
    version="1.0.0"
)

class ApplicantData(BaseModel):
    age_years: float = Field(..., gt=0, le=100, description="Age in years")
    years_employed: float = Field(..., ge=0, le=60, description="Years employed")
    amt_income_total: float = Field(..., gt=0, description="Annual income")
    amt_credit: float = Field(..., gt=0, description="Loan amount requested")
    amt_annuity: float = Field(..., gt=0, description="Monthly annuity payment")
    cnt_children: int = Field(..., ge=0, le=20, description="Number of children")

@app.get("/")
def root():
    return {"message": "Credit Risk Analyzer API is running", "version": "1.0.0"}

@app.post("/api/v1/predict")
def predict_credit_score(data: ApplicantData):
    features = build_features(
        data.age_years,
        data.years_employed,
        data.amt_income_total,
        data.amt_credit,
        data.amt_annuity,
        data.cnt_children
    )

    risk_probability = float(model.predict_proba(features)[0][1])
    credit_score = compute_score(risk_probability)
    decision = get_decision(risk_probability)

    return {
        "credit_score": credit_score,
        "risk_probability": round(risk_probability, 3),
        "decision": decision
    }