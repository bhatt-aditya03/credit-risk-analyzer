# 🏦 Alternative Credit Risk Analyzer

ML-based credit scoring system trained on **307,511 real loan applications**.  
Uses XGBoost to predict default risk and generate credit scores (300-900).

![App Screenshot](screenshot.png)

## 🚀 Live Links
- **Webapp:** https://credit-risk-analyzer-03.streamlit.app
- **API Docs:** https://credit-risk-analyzer-09qq.onrender.com/docs

## 🛠️ Tech Stack
Python, XGBoost, Scikit-learn, Pandas, FastAPI, Streamlit

## 📊 Model Performance
| Model | ROC-AUC |
|-------|---------|
| Logistic Regression (baseline) | 0.6063 |
| XGBoost | 0.6805 |

## 📁 Dataset
Uses the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset from Kaggle (307,511 loan applications, 122 features). Not included in repo due to size.

## ⚙️ How It Works
1. Cleans and engineers features from raw loan data
2. Trains XGBoost classifier to predict default probability
3. Maps probability to credit score: `score = 300 + (1 - risk) × 600`
4. API and webapp allow real-time scoring of new applicants

## 📂 Project Structure
```
├── app.py              # Streamlit webapp
├── main.py             # FastAPI backend
├── model.py            # Model training script
├── model_xgb.pkl       # Trained XGBoost model
├── notebooks/          # Exploratory analysis scripts
└── requirements.txt    # Dependencies
```