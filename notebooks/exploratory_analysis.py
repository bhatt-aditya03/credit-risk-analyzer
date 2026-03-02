applicant_name = "Aditya Bhatt"
applicant_age = 22
monthly_income = 25000.0
has_existing_loan = False

print("Applicant Name:", applicant_name)
print("Age:", applicant_age)
print("Monthly Income:", monthly_income)
print("Has Existing Loan:", has_existing_loan)

annual_income = monthly_income * 12
loan_amount = 200000
debt_to_income_ratio = loan_amount / annual_income

print("Annual Income:", annual_income)
print("Debt to Income Ratio:", debt_to_income_ratio)

if debt_to_income_ratio > 0.4:
    print("Loan Decision: HIGH RISK - Likely Rejected")

elif debt_to_income_ratio > 0.2:
    print("Loan Decision: MEDIUM RISK - Manual Review")

else:
    print("Loan Decision: LOW RISK - Likely Approved")


def calculate_credit_score(loan_amount, monthly_income):
    annual_income = monthly_income * 12
    ratio = loan_amount / annual_income

    if ratio > 0.4:
        risk_probability = 0.85
    elif ratio > 0.2:
        risk_probability = 0.45
    else:
        risk_probability = 0.15

    score = 300 + (1 - risk_probability) * 600
    return round(score)

applicants = [
    ("Aditya", 200000, 25000),
    ("Rahul", 50000, 40000),
    ("Priya", 500000, 30000)
]

for name, loan, income in applicants:
    score = calculate_credit_score(loan, income)
    print(f"{name} -> Credit Score: {score}")