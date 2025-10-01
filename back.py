from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score

app = Flask(__name__)

# ------------------- Load pre-trained artifacts -------------------
try:
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    explainer = shap.TreeExplainer(model)
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler/features: {e}")

# ------------------- Helper functions -------------------
def safe_numeric(series, fill=0):
    return pd.to_numeric(series, errors='coerce').fillna(fill)

def compute_features(df):
    # Doc sufficiency
    df['Doc_Sufficiency_%'] = ((df.get('Uploaded_Docs',0) / df.get('Required_Docs',1)) * 100).fillna(0)
    
    # NAAC Score mapping
    naac_map = {"A++":95,"A+":90,"A":85,"B++":75,"B+":70,"B":60,"C":50}
    if 'NAAC_Score' not in df.columns:
        df['NAAC_Score'] = df.get('NAAC_Grade', pd.Series('B')).map(naac_map)
    else:
        df['NAAC_Score'] = safe_numeric(df['NAAC_Score'], fill=70)
        df['NAAC_Score'] = df['NAAC_Score'].fillna(df.get('NAAC_Grade', pd.Series('B')).map(naac_map))
    
    # Faculty normalization
    faculty_min = df['Faculty_Count'].min() if 'Faculty_Count' in df.columns else 0
    faculty_range = max(df['Faculty_Count'].max() - faculty_min, 1) if 'Faculty_Count' in df.columns else 1
    df['Faculty_Norm'] = 100 * (df.get('Faculty_Count', 0) - faculty_min) / faculty_range
    
    infra_score = df.get('Infra_Score', pd.Series(0)).fillna(0)
    
    # Compliance Score
    df['Compliance_Score'] = 0.4*df['Faculty_Norm'] + 0.3*infra_score + 0.3*df['NAAC_Score'].fillna(70)
    
    # Financial Health
    df['Financial_Health_Score'] = df.get('Financial_Health_Score', df.get('Revenue_per_Student',0)).fillna(0)
    
    # Research Productivity
    df['Research_Productivity'] = df.get('Research_Productivity', pd.Series(0)).fillna(0)
    df['Outcome_Indicator'] = df.get('Outcome_Indicator', pd.Series(0)).fillna(0)
    df['Trust_Risk'] = df.get('Trust_Risk', pd.Series(0)).fillna(0)
    
    # Fill remaining NaNs for model features
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df[features] = df[features].fillna(0)
    
    return df

# ------------------- API -------------------
@app.route("/predict", methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error":"Please upload Excel/CSV file with key 'file'"}), 400
        
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Compute features
        df = compute_features(df)
        
        # Prepare model input
        X = df[features].astype(float)
        X_scaled = scaler.transform(X)
        
        # ML Predictions
        probs = model.predict_proba(X_scaled)[:,1]
        y_pred_class = (probs > 0.5).astype(int)
        df['AI_Readiness_Score'] = (probs*100).tolist()
        
        # Rule-based Approval logic
        df['Approval_Status'] = ((df['Compliance_Score'] > 65) & (df['Doc_Sufficiency_%'] > 75)).astype(int)
        
        # Metrics (if ground truth available)
        if 'Actual_Approval_Status' in df.columns:
            y_true = df['Actual_Approval_Status'].astype(int)
        else:
            y_true = df['Approval_Status']  # fallback
        
        mae = mean_absolute_error(y_true, probs)
        rmse = root_mean_squared_error(y_true, probs)
        precision = precision_score(y_true, y_pred_class, zero_division=0)
        recall = recall_score(y_true, y_pred_class, zero_division=0)
        f1 = f1_score(y_true, y_pred_class, zero_division=0)
        
        metrics = {
            "MAE": float(mae*1000),
            "RMSE": float(rmse*1000),
            "Precision": float(precision*100),
            "Recall": float(recall*100),
            "F1_Score": float(f1*100)
        }
        
        # SHAP Values
        try:
            shap_vals = explainer.shap_values(X_scaled)
            shap_list = [sv.tolist() if isinstance(sv, np.ndarray) else list(sv) for sv in shap_vals]
            df['SHAP_Values'] = shap_list
        except Exception:
            df['SHAP_Values'] = [[0.0]*len(features)]*len(df)
        
        # Flagged reasons
        flagged_list = []
        reasons_list = []
        for _, row in df.iterrows():
            reasons = []
            if row['Faculty_Count'] < 20:
                reasons.append("Low faculty count")
            if row['Doc_Sufficiency_%'] < 80:
                reasons.append("Insufficient documents")
            if row['Compliance_Score'] < 70:
                reasons.append("Low compliance")
            flagged_list.append(len(reasons)>0)
            reasons_list.append(reasons)
        df['Flagged'] = flagged_list
        df['Reasons_Flagged'] = reasons_list
        
        # Ensure all numeric types are JSON-serializable
        df = df.applymap(lambda x: float(x) if isinstance(x, (np.float32, np.float64, np.floating)) else x)
        
        return jsonify({
            "metrics": metrics,
            "data": df.to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

