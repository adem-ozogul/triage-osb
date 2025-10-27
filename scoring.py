import joblib
import numpy as np
import pandas as pd

# Modeli yükle
model = joblib.load("model.pkl")

# 🔧 Güvenli tahmin fonksiyonu
def compute_risk_category(df):
    # Modelin tanıdığı sütunlara göre hizalama
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        # Eğer model eski scikit-learn sürümüyle eğitildiyse
        expected_cols = [f"feat_{i}" for i in range(5)]
        df = df.reindex(columns=expected_cols, fill_value=0)

    # Olasılık hesapla
    prob = model.predict_proba(df)[0][1]

    # Risk kategorisi belirle
    if prob < 0.33:
        category = "low"
    elif prob < 0.66:
        category = "uncertain"
    else:
        category = "high"

    # Öznitelik katkılarını basitçe hesapla
    contrib = dict(zip(df.columns, df.iloc[0].values))
    return category, float(prob), contrib

# 🔍 Basit açıklama fonksiyonu
def explain_contributions(contrib):
    top_feats = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    explanation = {k: round(v, 3) for k, v in top_feats[:5]}
    return explanation
