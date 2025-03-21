from fastapi import FastAPI, Request
import pickle
import re
import pandas as pd
# import numpy as np

app = FastAPI()

# Load models and encoders
with open("loan_pred.pkl", "rb") as file1:
    model = pickle.load(file1)

with open("emp_title_enc.pkl", "rb") as file2:
    emp_title_enc = pickle.load(file2)

with open("title_enc.pkl", "rb") as file3:
    title_enc = pickle.load(file3)

with open("scaler.pkl", "rb") as file4:
    scaler = pickle.load(file4)

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


@app.get("/")
def home():
    return {"message": "Home page of Loan Approval App"}


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # Convert input JSON to DataFrame (best way to do it)
    dataframe = pd.DataFrame([data])

    # Encode categorical features
    encoded_data = encoding(dataframe)

    # Scaling data
    scaled_data = scaler.transform(
        encoded_data
    )  # Fix: Changed fit_tranfrom() to transform()

    # Making prediction
    result = model.predict(encoded_data)

    return {
        "Loan Status": result.tolist(),
        "encoded": encoded_data.to_dict(orient="records")[0],
        #  "Scaled Data": scaled_data.tolist()
    }  # Convert numpy array to list for JSON response
