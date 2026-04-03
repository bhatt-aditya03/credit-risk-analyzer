import numpy as np

def build_features(age_years, years_employed, amt_income_total,
                   amt_credit, amt_annuity, cnt_children):
    """
    Builds the 9-feature vector used by the XGBoost model.
    Returns a numpy array ready for model.predict_proba().
    """
    credit_income_ratio = amt_credit / amt_income_total
    annuity_income_ratio = amt_annuity / amt_income_total
    credit_term = amt_annuity / amt_credit

    return np.array([[
        age_years,
        years_employed,
        amt_income_total,
        amt_credit,
        amt_annuity,
        credit_income_ratio,
        annuity_income_ratio,
        credit_term,
        cnt_children
    ]])


def compute_score(risk_probability):
    """Maps default probability to a 300-900 credit score."""
    return int(300 + (1 - risk_probability) * 600)


def get_decision(risk_probability):
    """Returns loan decision based on risk threshold."""
    if risk_probability > 0.5:
        return "REJECTED"
    elif risk_probability > 0.3:
        return "MANUAL REVIEW"
    else:
        return "APPROVED"