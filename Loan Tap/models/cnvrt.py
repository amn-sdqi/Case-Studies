import joblib
import pickle

model_path = "Loan Tap/models/loan_pred.pkl"


with open(model_path, "rb") as file1:
    model = pickle.load(file1)
joblib.dump(model, "loan_pred.joblib")

with open("Loan Tap/models/emp_title_enc.pkl", "rb") as file2:
    emp_title_enc = pickle.load(file2)
joblib.dump(emp_title_enc, "emp_title_enc.joblib")


with open("Loan Tap/models/title_enc.pkl", "rb") as file3:
    title_enc = pickle.load(file3)
joblib.dump(title_enc, "title_enc.joblib")
