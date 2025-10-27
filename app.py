import streamlit as st
import pandas as pd
import numpy as np
import tempfile, os, random
from features import extract_video_features
from scoring import compute_risk_category, explain_contributions

st.set_page_config(page_title="TRIAGE-OSB+", page_icon="🧩", layout="centered")

st.title("🧠 TRIAGE-OSB+ | Early Autism Risk Screening")
st.markdown("""
Bu uygulama **video + kısa anket + klinik gözlem** verilerini kullanarak 
erken dönem otizm riskini değerlendirmek için geliştirilmiştir.
""")

video_file = st.file_uploader("🎥 Videoyu Yükleyin (MP4 / MOV)", type=["mp4","mov"], key="video_upload")
name_call = st.number_input("İsimle seslenme saniyesi (isteğe bağlı)", min_value=0.0, step=0.5, key="name_call")

st.subheader("🧾 Mini Davranış Anketi")
questions = {
    "q1": "İsmi söylendiğinde genelde bakar mı?",
    "q2": "İstediği şeyi parmağıyla işaret eder mi?",
    "q3": "Basit taklitleri yapar mı (el sallama vb.)?",
    "q4": "Sosyal gülümseme/göz teması var mı?",
    "q5": "Adıyla çağrıldığında ses çıkarır/cevap verir mi?"
}
options = ["Evet", "Bazen", "Hayır"]
responses = {k: st.selectbox(v, options, key=k) for k,v in questions.items()}

colA, colB = st.columns(2)
run_real = colA.button("🚀 Gerçek Analiz Et", key="real_button")
run_sim = colB.button("🧪 Simülasyon Modu", key="sim_button")

def run_analysis(feats, responses):
    mapping = {"Evet":1.0, "Bazen":0.5, "Hayır":0.0}
    row = feats.copy()
    for k,v in responses.items():
        row[f"survey_{k}"] = mapping[v]
    df = pd.DataFrame([row])
    cat, prob, contrib = compute_risk_category(df)
    st.success(f"📊 Tahmin edilen risk: **{cat.upper()}** (olasılık ≈ {prob:.2f})")
    st.json(explain_contributions(contrib))

if run_real:
    if not video_file:
        st.warning("Lütfen bir video dosyası yükleyin veya simülasyon modunu seçin.")
    else:
        with st.spinner("Video analiz ediliyor..."):
            suffix = os.path.splitext(video_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_file.read())
                video_path = tmp.name
            feats, meta = extract_video_features(video_path)
            run_analysis(feats, responses)
            os.remove(video_path)
elif run_sim:
    with st.spinner("Örnek veriler hazırlanıyor..."):
        feats = {f"feat_{i}": round(random.uniform(0,1),3) for i in range(5)}
        st.info("🧩 Bu sonuçlar yalnızca test amaçlıdır (örnek veri).")
        run_analysis(feats, responses)

st.markdown("---")
st.caption("Bu araç yalnızca araştırma ve erken tarama amaçlıdır; kesin tanı için uzman değerlendirmesi gerekir.")
