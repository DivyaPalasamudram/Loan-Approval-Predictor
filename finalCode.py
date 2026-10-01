# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import pandas as pd

# ---------------------------------
# LOAD MODEL AND SCALER
# ---------------------------------

with open("LoanApprovalModel.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Exact variables used by the trained model
model_columns = list(model.feature_names_in_)

# ---------------------------------
# TITLE
# ---------------------------------

st.markdown(
    """
    <h1 style="
        text-align: center;
        background-color: #ffcccc;
        padding: 10px;
        color: #cc0000;
    ">
    <b>Loan Approval Predictor</b>
    </h1>
    """,
    unsafe_allow_html=True
)

st.header("Enter Loan Applicant's Details")

# ---------------------------------
# NUMERIC INPUTS
# ---------------------------------

requested_loan_amount = st.number_input(
    "Requested Loan Amount",
    min_value=0.0,
    value=10000.0,
    step=1000.0
)

fico_score = st.slider(
    "FICO Score",
    min_value=300,
    max_value=850,
    value=650,
    step=1
)

monthly_gross_income = st.number_input(
    "Monthly Gross Income",
    min_value=0.0,
    value=5000.0,
    step=500.0
)

monthly_housing_payment = st.number_input(
    "Monthly Housing Payment",
    min_value=0.0,
    value=1500.0,
    step=100.0
)

bankruptcy = st.selectbox(
    "Ever Bankrupt or Foreclosed?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# ---------------------------------
# FIND EXACT CATEGORIES FROM MODEL
# ---------------------------------

def get_model_categories(prefix):
    return [
        col
        for col in model_columns
        if col.startswith(prefix + "_")
    ]

reason_columns = get_model_categories("Reason")
employment_status_columns = get_model_categories("Employment_Status")
employment_sector_columns = get_model_categories("Employment_Sector")
lender_columns = get_model_categories("Lender")

# Because training used drop_first=True,
# one original category is the reference category
# and does not have its own dummy column.

reason = st.selectbox(
    "Reason for Loan",
    ["Reference Category"] + reason_columns,
    format_func=lambda x:
        "Other / Reference Category"
        if x == "Reference Category"
        else x.replace("Reason_", "").replace("_", " ").title()
)

employment_status = st.selectbox(
    "Employment Status",
    ["Reference Category"] + employment_status_columns,
    format_func=lambda x:
        "Other / Reference Category"
        if x == "Reference Category"
        else x.replace("Employment_Status_", "").replace("_", " ").title()
)

employment_sector = st.selectbox(
    "Employment Sector",
    ["Reference Category"] + employment_sector_columns,
    format_func=lambda x:
        "Other / Reference Category"
        if x == "Reference Category"
        else x.replace("Employment_Sector_", "").replace("_", " ").title()
)

lender = st.selectbox(
    "Lender",
    ["Reference Category"] + lender_columns,
    format_func=lambda x:
        "Reference Lender"
        if x == "Reference Category"
        else x.replace("Lender_", "")
)

# ---------------------------------
# BUILD MODEL INPUT
# ---------------------------------

# Start with every model variable set to 0
input_data = pd.DataFrame(
    0,
    index=[0],
    columns=model_columns,
    dtype=float
)

# Numeric variables
input_data.loc[0, "Requested_Loan_Amount"] = requested_loan_amount
input_data.loc[0, "FICO_score"] = fico_score
input_data.loc[0, "Monthly_Gross_Income"] = monthly_gross_income
input_data.loc[0, "Monthly_Housing_Payment"] = monthly_housing_payment
input_data.loc[0, "Ever_Bankrupt_or_Foreclose"] = bankruptcy

# Set selected dummy variables to 1
if reason != "Reference Category":
    input_data.loc[0, reason] = 1

if employment_status != "Reference Category":
    input_data.loc[0, employment_status] = 1

if employment_sector != "Reference Category":
    input_data.loc[0, employment_sector] = 1

if lender != "Reference Category":
    input_data.loc[0, lender] = 1

# ---------------------------------
# PREDICTION
# ---------------------------------

if st.button("Evaluate Loan"):

    # Must scale the data because the model
    # was trained using scaled features
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success("The prediction is: **Approved ✅**")
    else:
        st.error("The prediction is: **Denied ❌**")

    st.write(
        f"Predicted Approval Probability: **{probability:.1%}**"
    )
