import joblib
import numpy as np

model = joblib.load("model.pkl")

def compute_risk_category(df):
    prob = model.predict_proba(df)[0][1]
    if prob < 0.33:
        cat = "low"
    elif prob < 0.66:
        cat = "uncertain"
    else:
        cat = "high"
    contributions = {col: float(val) for col, val in zip(df.columns, df.iloc[0].values)}
    return cat, float(prob), contributions

def explain_contributions(contrib_dict):
    sorted_items = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": k, "impact": v} for k, v in sorted_items]
