# train.py
from utils import read_csv_file, merge_data, preprocess, train_and_evaluate
import pprint
import os

# === EDIT THESE PATHS if your files are located elsewhere ===
BASE = r"C:\Users\VENKATA SURYA\OneDrive\Desktop\ONE LAST PRACTICE\Resume_Project\Health_Insurance_claim_Predictor"
CLAIMS_PATH = os.path.join(BASE, "claims.csv")
PATIENTS_PATH = os.path.join(BASE, "patients.csv")
PROVIDERS_PATH = os.path.join(BASE, "providers.csv")
# If you have a separate payments.csv and want to merge it explicitly, set PAYMENTS_PATH, else set to None
PAYMENTS_PATH = None  # os.path.join(BASE, "payments.csv")

# If auto-detection fails you can set label here (case-sensitive to actual column name):
LABEL_COLUMN = None   # e.g. "status" or "status_pay" ; set to None to auto-detect

def main():
    print("📂 Reading CSV files...")
    claims = read_csv_file(CLAIMS_PATH)
    patients = read_csv_file(PATIENTS_PATH)
    providers = read_csv_file(PROVIDERS_PATH)
    payments = read_csv_file(PAYMENTS_PATH) if PAYMENTS_PATH else None

    print("🔄 Merging data...")
    merged = merge_data(claims, patients, providers, payments)
    print(f"✅ Merged shape: {merged.shape}")

    print("⚙️ Preprocessing (auto-detecting label if not provided)...")
    try:
        X, y, artifacts = preprocess(merged, label_column=LABEL_COLUMN)
    except Exception as e:
        print("ERROR in preprocessing:", e)
        print("If label auto-detection failed, set LABEL_COLUMN variable in train.py to your target column name and re-run.")
        raise

    print("✅ Preprocessing complete.")
    print(f"Feature columns used ({len(artifacts['features'])}): {artifacts['features'][:10]}{'...' if len(artifacts['features'])>10 else ''}")
    print(f"Label column: {artifacts['label_col']}")
    print("Training model...")

    results = train_and_evaluate(X, y, artifacts, model_out_path="model.joblib", artifacts_out_path="artifacts.joblib")

    print("\nTraining complete. Evaluation:")
    pprint.pprint(results)
    print("\nSaved model -> model.joblib  and artifacts -> artifacts.joblib")

if __name__ == "__main__":
    main()
