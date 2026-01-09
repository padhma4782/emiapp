import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
import numpy as np

# ----------------------------------------------------
# MLflow Configuration
# ----------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

#ELIGIBILITY_MODEL_URI = "models:/EMIELIGIBILITY_XGB_Classifier@champion"
#EMI_MODEL_URI = "models:/MAX_EMI_XGB_Regressor@champion"

#ELIGIBILITY_MODEL_URI = mlflow.pyfunc.load_model("models/eligibility")
#EMI_MODEL_URI = mlflow.pyfunc.load_model("models/maxemi")

ELIGIBILITY_MODEL_PATH = "models/eligibility"
EMI_MODEL_PATH = "models/emi"

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------
@st.cache_resource
def load_models():
    eligibility_model = mlflow.pyfunc.load_model(ELIGIBILITY_MODEL_URI)
    emi_model = mlflow.pyfunc.load_model(EMI_MODEL_URI)
    return eligibility_model, emi_model

eligibility_model, emi_model = load_models()

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="EMI Decision Engine", layout="centered")
st.title("🏦 EMI Eligibility & Maximum EMI Predictor")
st.markdown("---")

# ----------------------------------------------------
# Applicant Inputs (MATCH xtrain)
# ----------------------------------------------------
st.subheader("👤 Applicant Details")

age = st.number_input("Age", 18, 70, 35)
monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 500000, 80000)
years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 6.0)
dependents = st.number_input("Number of Dependents", 0, 10, 1)

existing_loans = st.selectbox("Existing Loans", [0, 1])
credit_score = st.number_input("Credit Score", 300, 900, 750)

requested_amount = st.number_input("Requested Loan Amount (₹)", 50000, 5000000, 300000)
requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 36)

expenses = st.number_input("Monthly Expenses (₹)", 0, 300000, 25000)
emergency_fund = st.number_input(
    "Emergency Fund (₹)", 0, 5000000, 150000
)


st.subheader("🎓 Employment & Living Details")

education_enc = st.selectbox("Education Level", [0, 1, 2, 3, 4])
employment_type_enc = st.selectbox("Employment Type", [0, 1, 2])
company_type_enc = st.selectbox("Company Type", [0, 1, 2, 3, 4])
house_type_enc = st.selectbox("House Type", [0, 1, 2, 3])

st.subheader("📦 EMI Scenario")

emi_edu = st.checkbox("Education EMI")
emi_home = st.checkbox("Home Appliances EMI")
emi_personal = st.checkbox("Personal Loan EMI")
emi_vehicle = st.checkbox("Vehicle EMI")

# ----------------------------------------------------
# Feature Engineering (EXACT TRAINING LOGIC)
# ----------------------------------------------------
disposable_funds = monthly_salary - expenses

input_df = pd.DataFrame([{
    "age": age,
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "dependents": dependents,
    "existing_loans": existing_loans,
    "credit_score": credit_score,
    "emergency_fund": emergency_fund,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
    "education_enc": education_enc,
    "employment_type_enc": employment_type_enc,
    "company_type_enc": company_type_enc,
    "house_type_enc": house_type_enc,
    "emi_scenario_Education": int(emi_edu),
    "emi_scenario_Home Appliances": int(emi_home),
    "emi_scenario_Personal Loan": int(emi_personal),
    "emi_scenario_Vehicle": int(emi_vehicle),
    "expenses": expenses
    
}])

st.markdown("---")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
if st.button("🔍 Evaluate EMI"):

    eligibility_pred = eligibility_model.predict(input_df)[0]

    if eligibility_pred == 1:
        st.success("✅ Applicant is **ELIGIBLE** for EMI")

        predicted_emi = emi_model.predict(input_df)[0]
        predicted_emi = max(500, round(predicted_emi))

        st.subheader("💰 Maximum EMI Amount")
        st.success(f"₹ {predicted_emi:,.0f}")

    else:
        st.error("❌ Applicant is **NOT ELIGIBLE** for EMI")
        st.info("EMI amount prediction is skipped.")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.caption("📌 Powered by MLflow Model Registry | XGBoost | Streamlit")
