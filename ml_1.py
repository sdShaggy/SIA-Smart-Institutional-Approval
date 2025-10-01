import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score

df = pd.read_excel("institution_data.xlsx")

# Document sufficiency
df['Uploaded_Docs'] = df.get('Uploaded_Docs', pd.Series(0))
df['Required_Docs'] = df.get('Required_Docs', pd.Series(1))
df['Doc_Sufficiency_%'] = ((df['Uploaded_Docs'] / df['Required_Docs']) * 100)\
                            .replace([np.inf, -np.inf], 0).fillna(0)

# NAAC Score
naac_map = {"A++":95,"A+":90,"A":85,"B++":75,"B+":70,"B":60,"C":50}
if 'NAAC_Score' not in df.columns or df['NAAC_Score'].isnull().all():
    df['NAAC_Score'] = df.get('NAAC_Grade', pd.Series(np.nan)).map(naac_map).fillna(70)
else:
    df['NAAC_Score'] = df['NAAC_Score'].fillna(df.get('NAAC_Grade', pd.Series(np.nan)).map(naac_map).fillna(70))

# Faculty & Infra
df['Faculty_Count'] = df.get('Faculty_Count', pd.Series(0))
faculty_min = df['Faculty_Count'].min()
faculty_range = max(df['Faculty_Count'].max() - faculty_min, 1)
faculty_norm = 100 * (df['Faculty_Count'] - faculty_min) / faculty_range

df['Infra_Score'] = df.get('Infra_Score', pd.Series(0))
infra_score = df['Infra_Score'].fillna(df['Infra_Score'].mean() if df['Infra_Score'].notna().any() else 0)

df['Compliance_Score'] = 0.4 * faculty_norm + 0.3 * infra_score + 0.3 * df['NAAC_Score']

# Financial Health
df['Financial_Health_Score'] = df.get('Financial_Health_Score', df.get('Revenue_per_Student', pd.Series(0))).fillna(0)

# Use existing values directly from table
df['Research_Productivity'] = df.get('Research_Productivity', pd.Series(0)).fillna(0)
df['Outcome_Indicator'] = df.get('Outcome_Indicator', pd.Series(0)).fillna(0)
df['Trust_Risk'] = df.get('Trust_Risk', pd.Series(0)).fillna(0)
df['Quality_Index'] = df.get('Quality_Index', pd.Series(0)).fillna(0)

# Year column
df['Year'] = df.get('Year', pd.Series(2025)).fillna(2025)  # default to 2025 if missing

fill_defaults = {
    'Faculty_Count': df['Faculty_Count'].median(),
    'Infra_Score': df['Infra_Score'].mean(),
    'Uploaded_Docs': 0,
    'Required_Docs': 1,
    'Financial_Health_Score': 0,
    'Research_Productivity': 0,
    'Outcome_Indicator': 0,
    'Trust_Risk': 0,
    'Quality_Index': 0,
    'Year': 2025
}
df.fillna(fill_defaults, inplace=True)

df['Weighted_Score'] = 0.6 * df['Compliance_Score'] + 0.4 * df['Doc_Sufficiency_%']
df['Approval_Status'] = (df['Weighted_Score'] > 80).astype(int)

features = [
    'Year','Faculty_Count','Infra_Score','NAAC_Score','Doc_Sufficiency_%','Compliance_Score',
    'Quality_Index','Financial_Health_Score','Research_Productivity','Outcome_Indicator','Trust_Risk'
]

X = df[features].fillna(0)
y = df['Approval_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_scaled, y)

df['AI_Readiness_Score'] = model.predict_proba(X_scaled)[:,1]*100
y_pred_prob = df['AI_Readiness_Score']/100
y_pred_class = model.predict(X_scaled)

mae = mean_absolute_error(y, y_pred_prob) * 1000
rmse = root_mean_squared_error(y, y_pred_prob) * 1000
precision = precision_score(y, y_pred_class, zero_division=0)
recall = recall_score(y, y_pred_class, zero_division=0)
f1 = f1_score(y, y_pred_class, zero_division=0)

print("\nTraining complete!")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")

joblib.dump(model,  "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

print("\nModels, scaler, and features saved successfully!")
