import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.DataFrame([
    {"feat_0": 0.1, "feat_1": 0.2, "feat_2": 0.3, "feat_3": 0.1, "feat_4": 0.2, "label": 1},
    {"feat_0": 0.8, "feat_1": 0.9, "feat_2": 0.9, "feat_3": 0.8, "feat_4": 0.9, "label": 0},
    {"feat_0": 0.6, "feat_1": 0.4, "feat_2": 0.5, "feat_3": 0.6, "feat_4": 0.5, "label": 0}
])

X = data[[c for c in data.columns if c.startswith("feat_")]]
y = data["label"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Model kaydedildi: model.pkl")
