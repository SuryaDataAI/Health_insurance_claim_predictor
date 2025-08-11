# utils.py
import os
import joblib
import pandas as pd
import numpy as np

# Filenames (expect these in project root)
MODEL_FILE = "model.joblib"
ARTIFACTS_FILE = "artifacts.joblib"

def _load_joblib_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

def load_model_and_artifacts():
    """
    Returns (model, artifacts) where:
      - model is an sklearn-like estimator
      - artifacts is a dict possibly containing 'features','numeric_feats','cat_feats','encoder','scaler','label_encoder','label_col'
    This function handles both:
      - model.joblib containing model or dict with 'model' and 'artifacts'
      - separate artifacts.joblib
    """
    # Try load model file
    loaded_model_file = _load_joblib_safe(MODEL_FILE)
    model = None
    artifacts = None

    if loaded_model_file is None:
        raise FileNotFoundError(f"{MODEL_FILE} not found in project root.")

    # If model file is a dict and contains 'model'
    if isinstance(loaded_model_file, dict):
        # Common patterns: {"model": model, "artifacts": artifacts} or {"model": model}
        model = loaded_model_file.get("model") or loaded_model_file.get("clf") or loaded_model_file.get("estimator")
        # artifacts might be embedded
        if "artifacts" in loaded_model_file:
            artifacts = loaded_model_file["artifacts"]
    else:
        model = loaded_model_file

    # If separate artifacts file exists, load/override
    loaded_artifacts_file = _load_joblib_safe(ARTIFACTS_FILE)
    if loaded_artifacts_file is not None:
        # sometimes artifacts file is dict with keys
        artifacts = loaded_artifacts_file if isinstance(loaded_artifacts_file, dict) else None

    # If still no artifacts, try guess from model-file dict
    if artifacts is None and isinstance(loaded_model_file, dict):
        # maybe keys like 'encoder','scaler','features' present at top-level
        possible = {}
        for key in ("features", "numeric_feats", "cat_feats", "encoder", "scaler", "label_encoder", "label_col"):
            if key in loaded_model_file:
                possible[key] = loaded_model_file[key]
        if possible:
            artifacts = possible

    # final ensure artifacts dict
    if artifacts is None:
        artifacts = {}

    return model, artifacts

def _safe_to_numeric(series, fill=0.0):
    # Try numeric conversion; fallback leave as-is
    out = pd.to_numeric(series, errors="coerce")
    return out.fillna(fill)

def _encode_categorical_cols(X, cat_feats, artifacts):
    """
    Encode categories using saved encoder if available else use pandas category codes.
    encoder expected to be an sklearn OrdinalEncoder-like object saved in artifacts['encoder'].
    """
    if not cat_feats:
        return X
    enc = artifacts.get("encoder", None)
    # prepare dataframe of cat columns as strings
    cat_df = X[cat_feats].astype(str).fillna("missing")

    if enc is not None:
        try:
            # enc.transform expects 2D array
            enc_vals = enc.transform(cat_df)
            enc_df = pd.DataFrame(enc_vals, columns=cat_feats, index=X.index)
            # replace columns
            for c in cat_feats:
                X[c] = enc_df[c]
            return X
        except Exception:
            # fallback
            pass

    # fallback: convert to category codes per column
    for c in cat_feats:
        X[c] = cat_df[c].astype("category").cat.codes
    return X

def preprocess_for_model(df, artifacts):
    """
    Attempt to transform df into X matrix matching training features.
    artifacts expected keys (optional): features, numeric_feats, cat_feats, encoder, scaler
    - If artifacts['features'] exists we will attempt to select those columns (creating missing cols filled with 0s).
    - Convert numeric-like columns to numeric; encode categorical columns.
    """
    df_local = df.copy()

    # Basic conversions: strip whitespace from object columns
    for c in df_local.select_dtypes(include=["object"]).columns:
        df_local[c] = df_local[c].astype(str).str.strip()

    # create numeric features list
    numeric_feats = artifacts.get("numeric_feats", [])
    cat_feats = artifacts.get("cat_feats", [])

    # If no numeric_feats provided, infer numeric-like columns by dtype
    if not numeric_feats:
        # choose numeric dtypes or columns that look like amounts/age/count
        inferred = [c for c in df_local.columns if df_local[c].dtype.kind in "biufc"]
        # add likely numeric names
        extra = [c for c in df_local.columns if any(tok in c.lower() for tok in ["amount","age","count","paid","cost","charge","total","days"])]
        numeric_feats = list(dict.fromkeys(inferred + extra))

    # Convert numeric features
    for c in numeric_feats:
        if c in df_local.columns:
            df_local[c] = _safe_to_numeric(df_local[c], fill=0.0)

    # Ensure categorical candidate list exists
    if not cat_feats:
        # guess low cardinality object columns excluding ids
        cat_candidates = []
        for c in df_local.select_dtypes(include=["object"]).columns:
            lc = c.lower()
            if ("id" in lc and len(df_local[c].unique())>500) or any(tok in lc for tok in ["phone","address","zip","name"]):
                continue
            if df_local[c].nunique() <= 200:
                cat_candidates.append(c)
        cat_feats = cat_candidates[:6]

    # Build feature DataFrame
    features = artifacts.get("features", None)
    if features:
        # build X with those features, creating missing columns filled as 0 / 'missing'
        X = pd.DataFrame(index=df_local.index)
        for feat in features:
            if feat in df_local.columns:
                X[feat] = df_local[feat]
            else:
                # if feature looks numeric, fill 0.0 else 'missing'
                if any(tok in feat.lower() for tok in ["amount","age","count","paid","total","days","charge","cost"]):
                    X[feat] = 0.0
                else:
                    X[feat] = "missing"
        # Now convert numeric parts of X
        for c in X.columns:
            if c in numeric_feats:
                X[c] = _safe_to_numeric(X[c], fill=0.0)
    else:
        # No feature list — construct X from numeric_feats + cat_feats
        X = pd.DataFrame(index=df_local.index)
        for c in numeric_feats:
            if c in df_local.columns:
                X[c] = _safe_to_numeric(df_local[c], fill=0.0)
        for c in cat_feats:
            if c in df_local.columns:
                X[c] = df_local[c].astype(str).fillna("missing")
            else:
                X[c] = "missing"

    # Encode categorical columns if present in X
    present_cat_feats = [c for c in cat_feats if c in X.columns]
    if present_cat_feats:
        X = _encode_categorical_cols(X, present_cat_feats, artifacts)

    # Scaling numeric columns if scaler in artifacts
    scaler = artifacts.get("scaler", None)
    if scaler is not None:
        try:
            # apply scaler only to numeric columns intersection
            numeric_in_X = [c for c in X.columns if X[c].dtype.kind in "fi"]
            if numeric_in_X:
                X[numeric_in_X] = scaler.transform(X[numeric_in_X])
        except Exception:
            # ignore scaler errors
            pass

    # Final ensure numeric dtype where possible
    for c in X.columns:
        if X[c].dtype == object:
            # try convert
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X

def predict_from_csv(path, model, artifacts):
    """
    Return tuple (success, result)
     - success True => result is DataFrame with original columns + 'Prediction'
     - success False => result is error message string
    """
    if not os.path.exists(path):
        return False, f"CSV not found: {path}"
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return False, f"Failed to read CSV: {e}"

    try:
        X = preprocess_for_model(df, artifacts)
        # Ensure columns order matches artifacts['features'] if provided
        feat_list = artifacts.get("features", None)
        if feat_list:
            missing = [c for c in feat_list if c not in X.columns]
            # Add missing as zeros
            for m in missing:
                X[m] = 0.0
            X = X[feat_list]
        # predict
        preds = model.predict(X.values if hasattr(X, "values") else X)
        # If label encoder present, inverse transform
        label_encoder = artifacts.get("label_encoder", None)
        if label_encoder is not None:
            try:
                pred_labels = label_encoder.inverse_transform(preds)
            except Exception:
                pred_labels = preds
        else:
            # map common binary numeric labels to human friendly text if possible
            if set(np.unique(preds)).issubset({0,1}):
                pred_labels = np.where(preds==1, "Fraud", "Not Fraud")
            else:
                pred_labels = preds

        df_out = df.copy()
        df_out["Prediction"] = pred_labels
        return True, df_out

    except Exception as e:
        return False, f"Prediction failed: {e}"
