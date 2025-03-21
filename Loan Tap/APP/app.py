import os
import streamlit as st
import pickle
import re
import pandas as pd

# Load models and encoders
model_path = "/loan_pred.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload it.")
else:
    with open(model_path, "rb") as file1:
        model = pickle.load(file1)
    st.success("Model loaded successfully!")

with open("/emp_title_enc.pkl", "rb") as file2:
    emp_title_enc = pickle.load(file2)

with open("/title_enc.pkl", "rb") as file3:
    title_enc = pickle.load(file3)

# Define mappings
grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
sub_grade_mapping = {
    f"{grade}{num}": i + 1
    for i, (grade, num) in enumerate(
        [(g, n) for g in ["A", "B", "C", "D", "E", "F", "G"] for n in range(1, 6)]
    )
}
emp_length_map = {
    "10+ years": 10,
    "2 years": 2,
    "< 1 year": 0.5,
    "3 years": 3,
    "5 years": 5,
    "1 year": 1,
    "4 years": 4,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "Not Provided": 0.1,
}
home_ownership_map = {
    "ANY": 1,
    "MORTGAGE": 13,
    "NONE": 14,
    "OTHER": 15,
    "OWN": 15.1,
    "RENT": 18,
}
verification_status_map = {"Not Verified": 0, "Source Verified": 1, "Verified": 0.5}
purpose_map = {
    "car": 1,
    "credit_card": 1.1,
    "debt_consolidation": 2,
    "educational": 3,
    "home_improvement": 4,
    "house": 4.1,
    "major_purchase": 4.2,
    "medical": 4.3,
    "moving": 4.4,
    "other": 5,
    "renewable_energy": 6,
    "small_business": 7,
    "vacation": 8,
    "wedding": 9,
}
initial_list_status_map = {"f": 1, "w": 0}
application_type_map = {"DIRECT_PAY": 1, "INDIVIDUAL": 2, "JOINT": 3}


def encoding(df):
    df["term"] = df["term"].map({"36 months": 0, "60 months": 1})
    df["grade"] = df["grade"].map(grade_mapping)
    df["sub_grade"] = df["sub_grade"].map(sub_grade_mapping)
    df["emp_length"] = df["emp_length"].map(emp_length_map)
    df["home_ownership"] = df["home_ownership"].map(home_ownership_map)
    df["verification_status"] = df["verification_status"].map(verification_status_map)
    df["purpose"] = df["purpose"].map(purpose_map)
    df["initial_list_status"] = df["initial_list_status"].map(initial_list_status_map)
    df["application_type"] = df["application_type"].map(application_type_map)

    # Handle datetime features
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_d"] = df["issue_d"].apply(lambda x: (x.year * 10 + x.month) / 1000)

    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["earliest_cr_line"] = df["earliest_cr_line"].apply(
        lambda x: (x.year * 10 + x.month) / 1000
    )

    # Apply label encoders
    df["emp_title"] = emp_title_enc.transform(df[["emp_title"]])
    df["title"] = title_enc.transform(df[["title"]])

    # Extract zip code (assuming address has a zip code)
    df["address"] = (
        df["address"]
        .apply(
            lambda x: re.search(r"\b\d{5}\b", x).group()
            if re.search(r"\b\d{5}\b", x)
            else "0"
        )
        .astype(int)
    )

    return df


st.title("🏦 Loan Approval Prediction")

st.write("Enter loan details to predict approval status.")

# Input form
with st.form("loan_form"):
    loan_amnt = st.number_input("Loan Amount", value=10000.0)
    term = st.selectbox(
        "Term : The number of payments on the loan.", ["36 months", "60 months"]
    )
    int_rate = st.number_input("Interest Rate (%)", value=11.44)
    installment = st.number_input(
        "Installment : The monthly payment owed by the borrower if the loan originates.",
        value=329.48,
    )
    grade = st.selectbox(
        " LoanTap assigned loan grade", ["A", "B", "C", "D", "E", "F", "G"]
    )
    sub_grade = st.selectbox(
        "LoanTap assigned loan subgrade",
        ["1", "2", "3", "4", "5"],
    )
    emp_title = st.text_input("Employment of Borrower", "Marketing / Tech / Business")
    emp_length = st.selectbox(
        "Years in Employment",
        [
            "Not Provided",
            "< 1 year",
            "1 year",
            "2 years",
            "3 years",
            "4 years",
            "5 years",
            "6 years",
            "7 years",
            "8 years",
            "9 years",
            "10+ years",
        ],
    )
    home_ownership = st.selectbox(
        "Home Ownership Status", ["NONE", "MORTGAGE", "OTHER", "OWN", "RENT", "ANY"]
    )

    annual_inc = st.number_input("Annual Income", value=117000.0)
    verification_status = st.selectbox(
        "Verification Status", ["Verified", "Not Verified"]
    )
    issue_d = st.date_input(
        "Issue Date : The month which the loan was funded",
        value=None,
        format="YYYY-MM-DD",
    )
    purpose = st.selectbox(
        "Purpose",
        [
            "car",
            "credit_card",
            "debt_consolidation",
            "educational",
            "home_improvement",
            "house",
            "major_purchase",
            "medical",
            "moving",
            "other",
            "renewable_energy",
            "small_business",
            "vacation",
            "wedding",
        ],
    )
    title = st.text_input("The loan title provided by the borrower", "Personal loan ?")
    dti = st.number_input(
        "Debt-to-Income Ratio (DTI)", value=(installment / (annual_inc / 12)) * 100
    )
    earliest_cr_line = st.date_input(
        "Date : The month the borrower's earliest reported credit line was opened",
        value=None,
        format="YYYY-MM-DD",
    )
    open_acc = st.number_input(
        "Open Accounts : The number of open credit lines in the borrower's credit file.",
        value=16,
    )
    pub_rec = st.number_input(
        "Public Records : Number of derogatory public records", value=0
    )
    revol_bal = st.number_input("Total credit revolving balance", value=36369.0)
    revol_util = st.number_input(
        "Revolving Utilization (%) : Amount the borrower is using relative to all available revolving credit",
        value=41.8,
    )
    total_acc = st.number_input(
        "Total Accounts : The total number of credit lines currently in the borrower's credit file",
        value=25.0,
    )
    initial_list_status = st.selectbox(
        "initial listing status of the loan", {"Waiting": "w", "Fulfilled": "f"}
    )
    application_type = st.selectbox(
        "Application Type", ["INDIVIDUAL", "JOINT", "DIRECT_PAY"]
    )
    mort_acc = st.number_input("Number of mortgage accounts", value=0.0)
    pub_rec_bankruptcies = st.selectbox(
        "Public Record Bankruptcies", {"Yes": 1.0, "No": 0.0}
    )
    address = st.text_input("Address", "0174 Michelle GatewayMendozaberg, OK 22690")

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    # Convert form inputs to JSON
    data = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": grade + sub_grade,
        "emp_title": emp_title,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "issue_d": issue_d,
        "purpose": purpose,
        "title": title,
        "dti": dti,
        "earliest_cr_line": earliest_cr_line,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "initial_list_status": "w" if initial_list_status == "Waiting" else "f",
        "application_type": application_type,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": 1.0 if pub_rec_bankruptcies == "Yes" else 0.0,
        "address": address,
    }

    # Convert input JSON to DataFrame (best way to do it)
    dataframe = pd.DataFrame([data])

    # Encode categorical features
    encoded_data = encoding(dataframe)
    
    # st.write(encoded_data)
    # Making prediction
    result = model.predict(encoded_data)
    
    Status= "Your Loan has been Approved" if result==1 else "Sorry Our System has suggested you for rejection"
    
    st.write(Status)
