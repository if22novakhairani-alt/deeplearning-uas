import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.title("Prediksi Risiko Penyakit Jantung")

# ===============================
# Load model dan scaler
# ===============================
model = load_model("model_ann.keras")
scaler = joblib.load("scaler.save")

# ===============================
# Input pengguna
# ===============================
age = st.slider("Umur (tahun)", 30, 80, 40)
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
height = st.number_input("Tinggi Badan (cm)", 100, 250, 165)
weight = st.number_input("Berat Badan (kg)", 30, 200, 70)
ap_hi = st.number_input("Tekanan Darah Sistolik (ap_hi)", 80, 200, 120)
ap_lo = st.number_input("Tekanan Darah Diastolik (ap_lo)", 50, 150, 80)
chol = st.selectbox("Kolesterol", ["Normal", "Tinggi", "Sangat Tinggi"])
gluc = st.selectbox("Glukosa", ["Normal", "Tinggi", "Sangat Tinggi"])

st.markdown("### Gaya Hidup")
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
# Feature engineering
# ===============================
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
mean_arterial_pressure = (ap_hi + 2 * ap_lo) / 3

# ===============================
# DataFrame untuk scaler
# ===============================
data = pd.DataFrame([{
    'age_years': age,
    'gender': gender_num,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': chol_num,
    'gluc': gluc_num,
    'smoke': int(smoke),
    'alco': int(alco),
    'active': int(active),
    'bmi': bmi,
    'pulse_pressure': pulse_pressure,
    'mean_arterial_pressure': mean_arterial_pressure
}])

# ===============================
# Prediksi
# ===============================
if st.button("Prediksi"):
    data_scaled = scaler.transform(data)  # scaling sesuai training
    prob = model.predict(data_scaled)[0][0]
    hasil = 1 if prob >= 0.5 else 0

    if hasil == 1:
        st.error(f"Hasil: Risiko Penyakit Jantung ({prob:.2%})")
    else:
        st.success(f"Hasil: Sehat ({(1 - prob):.2%})")
