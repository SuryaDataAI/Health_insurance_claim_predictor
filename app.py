# app.py
from flask import Flask, render_template
import os
from utils import load_model_and_artifacts, predict_from_csv

app = Flask(__name__, template_folder="templates")

# CSV absolute paths you provided earlier
CSV_FILES = {
    "Claims Data": r"C:\Users\VENKATA SURYA\OneDrive\Desktop\ONE LAST PRACTICE\Resume_Project\Health_Insurance_claim_Predictor\claims.csv",
    "Patients Data": r"C:\Users\VENKATA SURYA\OneDrive\Desktop\ONE LAST PRACTICE\Resume_Project\Health_Insurance_claim_Predictor\patients.csv",
    "Payments Data": r"C:\Users\VENKATA SURYA\OneDrive\Desktop\ONE LAST PRACTICE\Resume_Project\Health_Insurance_claim_Predictor\payments.csv",
    "Providers Data": r"C:\Users\VENKATA SURYA\OneDrive\Desktop\ONE LAST PRACTICE\Resume_Project\Health_Insurance_claim_Predictor\providers.csv"
}

# Load model & artifacts at startup
try:
    MODEL, ARTIFACTS = load_model_and_artifacts()
except Exception as e:
    MODEL, ARTIFACTS = None, {}
    print("Error loading model/artifacts:", e)

@app.route("/")
def home():
    # simple landing page with button to run predictions
    return render_template("index.html")

@app.route("/results")
def results():
    if MODEL is None:
        return render_template("error.html", message="Model not loaded. Check model.joblib in project root.")

    results = {}
    for name, path in CSV_FILES.items():
        ok, res = predict_from_csv(path, MODEL, ARTIFACTS)
        if not ok:
            results[name] = {"error": res}
        else:
            # prepare small summary + HTML table (first 20 rows)
            df = res
            total = len(df)
            # count fraud if labels equal "Fraud"/"Not Fraud" or numeric
            if "Prediction" in df.columns:
                counts = df["Prediction"].value_counts(dropna=False).to_dict()
            else:
                counts = {}

            table_html = df.head(20).to_html(classes="table table-dark table-striped", index=False, escape=False)
            results[name] = {
                "table": table_html,
                "total": total,
                "counts": counts
            }

    return render_template("result.html", results=results)

if __name__ == "__main__":
    # Set host=0.0.0.0 if you want external access
    app.run(debug=True)
