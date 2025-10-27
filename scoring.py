import random

def compute_risk_category(df):
    """
    Dummy risk calculator for simulation.
    Returns: category, probability, contributions
    """
    # Ortalama özelliğe göre basit puanlama
    prob = df.mean(axis=1).iloc[0]
    if prob < 0.33:
        cat = "low"
    elif prob < 0.66:
        cat = "uncertain"
    else:
        cat = "high"
    # Özellik katkıları (rastgele)
    contributions = {col: round(random.uniform(-0.3, 0.3), 3) for col in df.columns}
    return cat, float(prob), contributions


def explain_contributions(contrib_dict):
    """
    Convert contribution dict into sorted readable list.
    """
    sorted_items = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    explain = [{"feature": k, "impact": v} for k, v in sorted_items[:8]]
    return explain
