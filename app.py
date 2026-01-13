import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prediksi Risiko Jantung", layout="centered")
st.title("Prediksi Risiko Penyakit Jantung ❤️")

# ===============================
# Load model dan scaler
# ===============================
model = load_model("model_ann_full_features.keras")
scaler = joblib.load("scaler_full_features.save")

# ===============================
# Input pengguna
# ===============================
st.header("Data Pribadi")
age = st.slider("Umur (tahun)", 30, 80, 40)
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
height = st.number_input("Tinggi Badan (cm)", 100, 250, 165)
weight = st.number_input("Berat Badan (kg)", 30, 200, 70)

st.header("Tekanan Darah & Lab")
ap_hi = st.number_input("Tekanan Darah Sistolik (ap_hi)", 80, 200, 120)
ap_lo = st.number_input("Tekanan Darah Diastolik (ap_lo)", 50, 150, 80)
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
# Feature engineering otomatis
# ===============================
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
mean_arterial_pressure = (ap_hi + 2 * ap_lo) / 3

# ===============================
# Buat DataFrame sesuai urutan fitur scaler
# ===============================
feature_names = [
    'age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'bmi', 'pulse_pressure', 'mean_arterial_pressure'
]

data = pd.DataFrame([[
    age, gender_num, height, weight, ap_hi, ap_lo,
    chol_num, gluc_num, int(smoke), int(alco), int(active),
    bmi, pulse_pressure, mean_arterial_pressure
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
