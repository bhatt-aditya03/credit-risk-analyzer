import streamlit as st
import joblib
from pathlib import Path
from features import build_features, compute_score, get_decision

MODEL_PATH = Path(__file__).parent / "model_xgb.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦")

st.title("🏦 Credit Risk Analyzer")
st.write("Enter applicant details to get a credit score and loan decision.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=70, value=35)
    years_employed = st.number_input("Years Employed", min_value=0.0, max_value=50.0, value=5.0)
    income = st.number_input("Annual Income (₹)", min_value=10000, max_value=10000000, value=500000)

with col2:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=200000)
    annuity = st.number_input("Monthly Annuity (₹)", min_value=1000, max_value=500000, value=10000)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

if st.button("Analyze Credit Risk", type="primary"):
    features = build_features(age, years_employed, income, loan_amount, annuity, children)
    risk = float(model.predict_proba(features)[0][1])
    score = compute_score(risk)
    decision = get_decision(risk)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Credit Score", score)
    col2.metric("Risk Probability", f"{risk:.1%}")
    col3.metric("Decision", decision)

    if decision == "APPROVED":
        st.success("✅ Loan Approved")
    elif decision == "MANUAL REVIEW":
        st.warning("⚠️ Manual Review Required")
    else:
        st.error("❌ Loan Rejected")