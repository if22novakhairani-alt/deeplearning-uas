import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.caption("Aplikasi berbasis Deep Learning untuk estimasi risiko penyakit jantung")

# ===============================
# Load model & scaler (11 fitur)
# ===============================
model = load_model("model_ann_11features.keras")
scaler = joblib.load("scaler_11features.save")

# ===============================
# INPUT USER
# ===============================
st.header("📋 Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Umur (tahun)", 30, 80, 40)
    height = st.number_input("Tinggi Badan (cm)", 140, 210, 165)
    weight = st.number_input("Berat Badan (kg)", 40, 200, 70)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])

with col2:
    ap_hi = st.number_input("Tekanan Darah Sistolik", 80, 200, 120)
    ap_lo = st.number_input("Tekanan Darah Diastolik", 50, 150, 80)
    chol = st.selectbox("Kolesterol", ["Normal", "Tinggi", "Sangat Tinggi"])
    gluc = st.selectbox("Glukosa", ["Normal", "Tinggi", "Sangat Tinggi"])

st.subheader("🏃 Gaya Hidup")
smoke = st.checkbox("Merokok")
alco = st.checkbox("Konsumsi Alkohol")
active = st.checkbox("Aktif secara fisik")

# ===============================
# KONVERSI KATEGORIK → NUMERIK
# ===============================
gender_num = 1 if gender == "Perempuan" else 2
chol_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[chol]
gluc_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[gluc]

# ===============================
# DATAFRAME INPUT MODEL (11 fitur)
# ===============================
feature_names = [
    'age_years', 'height', 'weight',
    'ap_hi', 'ap_lo', 'gender',
    'cholesterol', 'gluc',
    'smoke', 'alco', 'active'
]

data = pd.DataFrame([[
    age, height, weight,
    ap_hi, ap_lo, gender_num,
    chol_num, gluc_num,
    int(smoke), int(alco), int(active)
]], columns=feature_names)

# ===============================
# PREDIKSI
# ===============================
st.divider()

if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # Scaling
    data_scaled = scaler.transform(data.values)
    prob = model.predict(data_scaled)[0][0]

    # ===============================
    # KLASIFIKASI LEVEL RISIKO
    # ===============================
    if prob < 0.4:
        level = "Rendah"
        color = "green"
    elif prob < 0.7:
        level = "Sedang"
        color = "orange"
    else:
        level = "Tinggi"
        color = "red"

    st.subheader("📊 Hasil Prediksi")

    st.progress(int(prob * 100))
    st.markdown(
        f"<h3 style='color:{color}'>Risiko {level} ({prob:.1%})</h3>",
        unsafe_allow_html=True
    )

    # ===============================
    # ANALISIS FAKTOR RISIKO
    # ===============================
    st.subheader("⚠️ Faktor Risiko Terdeteksi")

    risk_factors = []

    if ap_hi >= 140 or ap_lo >= 90:
        risk_factors.append("Tekanan darah tinggi")

    if chol_num >= 2:
        risk_factors.append("Kolesterol tinggi")

    if gluc_num >= 2:
        risk_factors.append("Glukosa darah tinggi")

    if smoke:
        risk_factors.append("Merokok")

    if not active:
        risk_factors.append("Kurang aktivitas fisik")

    if risk_factors:
        for rf in risk_factors:
            st.warning(f"• {rf}")
    else:
        st.success("Tidak ditemukan faktor risiko utama")

    # ===============================
    # REKOMENDASI OTOMATIS
    # ===============================
    st.subheader("💡 Rekomendasi Kesehatan")

    if "Merokok" in risk_factors:
        st.info("🚭 Disarankan untuk berhenti merokok")

    if "Kolesterol tinggi" in risk_factors:
        st.info("🥗 Kurangi konsumsi makanan berlemak")

    if "Tekanan darah tinggi" in risk_factors:
        st.info("🩺 Rutin memantau tekanan darah")

    if "Kurang aktivitas fisik" in risk_factors:
        st.info("🏃 Lakukan olahraga ringan secara rutin")

    if not risk_factors:
        st.info("👍 Pertahankan gaya hidup sehat Anda")

    # ===============================
    # DETAIL INPUT (EXPLAINABILITY)
    # ===============================
    with st.expander("📌 Detail Data yang Digunakan Model"):
        st.dataframe(data)
