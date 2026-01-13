import app as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# ===============================
# Load model deep learning
# ===============================
model = load_model("model_jantung_ann.keras", compile=False)  # sesuaikan nama file

st.title("Prediksi Risiko Penyakit Jantung")

# ===============================
# Input pengguna
# ===============================
age = st.slider("Umur (dalam tahun)", 30, 80, 40)
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Tekanan Darah Sistolik (ap_hi)", value=120)
ap_lo = st.number_input("Tekanan Darah Diastolik (ap_lo)", value=80)
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
# Feature engineering (TIDAK DIUBAH)
# ===============================
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
mean_arterial_pressure = (ap_hi + 2 * ap_lo) / 3

# ===============================
# DataFrame (sesuai fitur training)
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
    'smoke': int(bool(smoke)),
    'alco': int(bool(alco)),
    'active': int(bool(active)),
    'bmi': bmi,
    'pulse_pressure': pulse_pressure,
    'mean_arterial_pressure': mean_arterial_pressure
}])

# ===============================
# Prediksi
# ===============================
if st.button("Prediksi"):
    prob = model.predict(data)[0][0]   # probabilitas kelas 1
    hasil = 1 if prob >= 0.5 else 0

    if hasil == 1:
        st.error(f"Hasil: Risiko Penyakit Jantung ({prob:.2%})")
    else:

        st.success(f"Hasil: Sehat ({(1 - prob):.2%})")
