import streamlit as st
import pandas as pd
import numpy as np
import tempfile, os, random
from features import extract_video_features
from scoring import compute_risk_category, explain_contributions

# 🌈 Sayfa ayarları
st.set_page_config(
    page_title="TRIAGE-OSB+",
    page_icon="🧩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 🌟 Başlık alanı
st.markdown(
    """
    <div style="background-color:#e3f2fd;padding:15px;border-radius:12px;text-align:center;">
        <h1 style="color:#1565c0;">🧠 TRIAGE-OSB+</h1>
        <p style="color:#424242;font-size:17px;">
        <b>Video + Anket + Klinik gözlem</b> tabanlı erken otizm riski değerlendirme sistemi.<br>
        Bu araç araştırma ve erken tarama amacıyla geliştirilmiştir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 📂 Video yükleme
st.markdown("### 🎥 Video Yükleme")
video_file = st.file_uploader("MP4 veya MOV formatında video seçin", type=["mp4", "mov"], key="video_upload")
st.caption("⚠️ Video yalnızca analiz için geçici olarak işlenir, sunucuya kaydedilmez.")

# 📋 Mini davranış anketi
st.markdown("### 🧾 Davranış Anketi")
st.markdown(
    "<small>Lütfen çocuğun tipik davranışlarını temel alarak yanıtlayın:</small>",
    unsafe_allow_html=True,
)

questions = {
    "q1": "İsmi söylendiğinde genelde bakar mı?",
    "q2": "İstediği şeyi parmağıyla işaret eder mi?",
    "q3": "Basit taklitleri yapar mı (el sallama vb.)?",
    "q4": "Sosyal gülümseme/göz teması var mı?",
    "q5": "Adıyla çağrıldığında ses çıkarır/cevap verir mi?",
}
options = ["Evet", "Bazen", "Hayır"]
responses = {k: st.selectbox(v, options, key=k) for k, v in questions.items()}

name_call = st.number_input("İsimle seslenme süresi (saniye)", min_value=0.0, step=0.5, key="name_call")

st.markdown("---")

# 🔘 İşlem butonları
colA, colB = st.columns(2)
run_real = colA.button("🚀 Gerçek Analiz Et", use_container_width=True)
run_sim = colB.button("🧪 Simülasyon Modu", use_container_width=True)

# 🔍 Analiz fonksiyonu
def run_analysis(feats, responses):
    mapping = {"Evet": 1.0, "Bazen": 0.5, "Hayır": 0.0}
    row = feats.copy()
    for k, v in responses.items():
        row[f"survey_{k}"] = mapping[v]
    df = pd.DataFrame([row])
    cat, prob, contrib = compute_risk_category(df)

    color = {"low": "#81c784", "uncertain": "#fff176", "high": "#e57373"}[cat]
    st.markdown(
        f"""
        <div style="background-color:{color};padding:15px;border-radius:10px;text-align:center;">
            <h3>📊 Tahmin Edilen Risk: <b>{cat.upper()}</b></h3>
            <p>Olasılık ≈ <b>{prob:.2f}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔍 Model Katkı Analizi")
    st.json(explain_contributions(contrib))

# 🚀 Gerçek analiz
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

# 🧪 Simülasyon
elif run_sim:
    with st.spinner("Örnek veriler hazırlanıyor..."):
        feats = {f"feat_{i}": round(random.uniform(0, 1), 3) for i in range(5)}
        st.info("🧩 Bu sonuçlar yalnızca test amaçlıdır (örnek veri).")
        run_analysis(feats, responses)

# 📘 Alt bilgi
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:13px;'>© 2025 TRIAGE-OSB+ | Klinik araştırma ve erken tarama amacıyla geliştirilmiştir.</p>",
    unsafe_allow_html=True,
)
