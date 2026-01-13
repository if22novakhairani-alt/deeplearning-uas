import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Page config & title
# ===============================
st.set_page_config(page_title="Prediksi Risiko Jantung", layout="centered")
st.title("Prediksi Risiko Penyakit Jantung")

# ===============================
# Load model & scaler baru (11 fitur)
# ===============================
model = load_model("model_ann_dl_full.keras")  # model hasil training 11 fitur
scaler = joblib.load("scaler_dl.save")   # scaler untuk 11 fitur

# ===============================
# Input pengguna
# ===============================
st.header("Data Pribadi")
age = st.slider("Umur (tahun)", 30, 80, 40)
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
height = st.number_input("Tinggi Badan (cm)", 100, 250, 165)
weight = st.number_input("Berat Badan (kg)", 30, 200, 70)

st.header("Tekanan Darah & Lab")
ap_hi = st.number_input("Tekanan Darah Sistolik", 80, 200, 120)
ap_lo = st.number_input("Tekanan Darah Diastolik", 50, 150, 80)
chol = st.selectbox("Kolesterol", ["Normal", "Tinggi", "Sangat Tinggi"])
gluc = st.selectbox("Glukosa", ["Normal", "Tinggi", "Sangat Tinggi"])

st.header("Gaya Hidup")
smoke = st.checkbox("Merokok")
alco = st.checkbox("Konsumsi Alkohol")
active = st.checkbox("Aktif secara fisik")

# ===============================
# Konversi input ke numerik
# ===============================
gender_num = 1 if gender == "Perempuan" else 2
chol_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[chol]
gluc_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[gluc]

# ===============================
# Buat DataFrame sesuai urutan fitur scaler (11 fitur)
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
# Prediksi
# ===============================
if st.button("Prediksi"):
    data_scaled = scaler.transform(data.values)
    prob = model.predict(data_scaled)[0][0]
    hasil = 1 if prob >= 0.5 else 0

    st.markdown("---")
    if hasil == 1:
        st.error(f"Hasil: Risiko Penyakit Jantung ({prob:.2%})")
    else:
        st.success(f"Hasil: Sehat ({(1 - prob):.2%})")
