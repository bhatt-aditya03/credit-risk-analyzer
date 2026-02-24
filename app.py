import streamlit as st
import joblib
import numpy as np

model = joblib.load("model_xgb.pkl")

st.set_page_config(page_title = "Credit Risk Analyzer", page_icon = "🏦")

st.title("🏦 Credit Risk Analyzer")
st.write("Enter applicant details to get a credit score and loan decision.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age(years)", min_value = 18, max_value = 70, value = 35)
    years_employed = st.number_input("Years Employed", min_value = 0.0, max_value = 50.0, value = 5.0)
    income = st.number_input("Annual Income (₹)", min_value = 10000, max_value = 10000000, value = 500000)

with col2:
    loan_amount = st.number_input("Loan Amount (₹)", min_value = 10000, max_value = 10000000, value = 200000)
    annuity = st.number_input("Monthly Annuity (₹)", min_value = 1000, max_value = 500000, value = 150000)
    children = st.number_input("Number of Children", min_value = 0, max_value = 10, value = 0)

if st.button("Analyze Credit Risk", type = "primary"):
    credit_income_ratio = loan_amount / income
    annuity_income_ratio = annuity / income
    credit_term = annuity / loan_amount

    features = np.array([[
        age, years_employed, income, loan_amount, annuity,
        credit_income_ratio, annuity_income_ratio, credit_term, children
    ]])

    risk = model.predict_proba(features)[0][1]
    score = int(300 + (1 - risk) * 600)

    if risk > 0.5:
        decision = "REJECTED"
        color = "red"
    elif risk > 0.3:
        decision = "MANUAL REVIEW"
        color = "orange"
    else:
        decision = "APPROVED"
        color = "green"

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