import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load Model & Transformer Files
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl") # Loading the OneHotEncoder saved from the notebook
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🏦 CreditWise Loan Approval Predictor")
st.write("Enter applicant details to check loan eligibility.")
st.divider()

# -----------------------------
# Sliders & Numeric Inputs
# -----------------------------
st.subheader("Financial & Personal Details")
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 70, 30)
    Applicant_Income = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=5000)
    Coapplicant_Income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=5000)
    Loan_Amount = st.number_input("Loan Amount (₹)", min_value=1000, value=20000, step=1000)
    Loan_Term = st.slider("Loan Term (Months)", 12, 120, 48, step=12)
    Credit_Score = st.slider("Credit Score", 300, 900, 650)

with col2:
    Dependents = st.slider("Dependents", 0, 10, 0)
    Existing_Loans = st.slider("Existing Loans", 0, 10, 0)
    DTI_Ratio = st.slider("Debt To Income Ratio", 0.0, 1.0, 0.3, step=0.01)
    Savings = st.number_input("Savings (₹)", min_value=0, value=10000, step=1000)
    Collateral_Value = st.number_input("Collateral Value (₹)", min_value=0, value=20000, step=1000)

st.divider()

# -----------------------------
# Dropdown Inputs
# -----------------------------
st.subheader("Categorical Details")
col3, col4 = st.columns(2)

with col3:
    Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    Marital_Status = st.selectbox("Marital Status", ["Single", "Married"])
    Gender = st.selectbox("Gender", ["Male", "Female"])

with col4:
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    Loan_Purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Car", "Business"])
    Employer_Category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])

st.divider()

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Loan Approval"):

    # 1. Recreate Feature Engineering from notebook
    DTI_Ratio_sq = DTI_Ratio ** 2
    Credit_Score_sq = Credit_Score ** 2

    # 2. Recreate LabelEncoder logic for Education manually (Alphabetical: Graduate=0, Not Graduate=1)
    edu_encoded = 1 if Education_Level == "Not Graduate" else 0

    # 3. Create DataFrame for OneHotEncoder Categorical variables
    cat_data = pd.DataFrame({
        "Employment_Status": [Employment_Status],
        "Marital_Status": [Marital_Status],
        "Loan_Purpose": [Loan_Purpose],
        "Property_Area": [Property_Area],
        "Gender": [Gender],
        "Employer_Category": [Employer_Category]
    })

    # Apply the saved OneHotEncoder
    encoded_cats = ohe.transform(cat_data)
    cat_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_data.columns))

    # 4. Create DataFrame for Numeric variables (matching notebook structure before scaling)
    num_data = pd.DataFrame({
        "Applicant_Income": [Applicant_Income],
        "Coapplicant_Income": [Coapplicant_Income],
        "Age": [Age],
        "Dependents": [Dependents],
        "Existing_Loans": [Existing_Loans],
        "Savings": [Savings],
        "Collateral_Value": [Collateral_Value],
        "Loan_Amount": [Loan_Amount],
        "Loan_Term": [Loan_Term],
        "Education_Level": [edu_encoded],
        "DTI_Ratio_sq": [DTI_Ratio_sq],
        "Credit_Score_sq": [Credit_Score_sq]
    })

    # 5. Combine Numeric and Categorical into one row
    final_input = pd.concat([num_data, cat_df], axis=1)

    # 6. Ensure column order exactly matches what the model was trained on
    final_input = final_input.reindex(columns=model_columns, fill_value=0)

    # 7. Scale the input using the saved StandardScaler
    input_scaled = scaler.transform(final_input)

    # 8. Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    approval_prob = probability[0][1]

    st.write(f"### Approval Probability: {round(approval_prob * 100, 2)}%")

    if prediction[0] == 1:
        st.success("✅ **Loan Approved**")
    else:
        st.error("❌ **Loan Not Approved**")