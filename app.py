import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from tensorflow.keras.models import load_model

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    layout="centered"
)

st.title("Prediksi Risiko Penyakit Jantung")
st.caption("Aplikasi berbasis Deep Learning dengan riwayat pasien")

# ===============================
# Load model & scaler
# ===============================
model = load_model("model_ann_dl_full.keras")
scaler = joblib.load("scaler_dl.save")

# ===============================
# File histori
# ===============================
HISTORY_FILE = "riwayat_pasien.csv"

# ===============================
# INPUT USER
# ===============================
st.header("Data Pasien")

nama = st.text_input("Nama Pasien", placeholder="Masukkan nama lengkap")

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

st.subheader("Gaya Hidup")
smoke = st.checkbox("Merokok")
alco = st.checkbox("Konsumsi Alkohol")
active = st.checkbox("Aktif secara fisik")

# ===============================
# Konversi ke numerik
# ===============================
gender_num = 1 if gender == "Perempuan" else 2
chol_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[chol]
gluc_num = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[gluc]

# ===============================
# Data untuk model (11 fitur)
# ===============================
feature_names = [
    'age_years', 'height', 'weight',
    'ap_hi', 'ap_lo', 'gender',
    'cholesterol', 'gluc',
    'smoke', 'alco', 'active'
]

data_model = pd.DataFrame([[
    age, height, weight,
    ap_hi, ap_lo, gender_num,
    chol_num, gluc_num,
    int(smoke), int(alco), int(active)
]], columns=feature_names)

# ===============================
# PREDIKSI
# ===============================
st.divider()

if st.button("Prediksi Risiko", use_container_width=True):

    if nama.strip() == "":
        st.error("Nama pasien wajib diisi")
        st.stop()

    # Predict
    data_scaled = scaler.transform(data_model.values)
    prob = model.predict(data_scaled)[0][0]

    # Risk level
    if prob < 0.4:
        level = "Rendah"
    elif prob < 0.7:
        level = "Sedang"
    else:
        level = "Tinggi"

    # ===============================
    # SIMPAN KE HISTORI
    # ===============================
    record = {
        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama": nama,
        "Umur": age,
        "Jenis Kelamin": gender,
        "Risiko (%)": round(prob * 100, 2),
        "Level Risiko": level
    }

    df_new = pd.DataFrame([record])

    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(HISTORY_FILE, index=False)

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader(f"Hasil Prediksi untuk {nama}")
    st.progress(int(prob * 100))
    st.success(f"Risiko {level} ({prob:.1%})")

# ===============================
# HISTORI PASIEN
# ===============================
st.divider()
st.header("Riwayat Pasien")

if os.path.exists(HISTORY_FILE):
    df_history = pd.read_csv(HISTORY_FILE)

    st.dataframe(df_history, use_container_width=True)

    colA, colB = st.columns(2)

    # Hapus satu data
    with colA:
        idx_delete = st.number_input(
            "Hapus data berdasarkan nomor baris",
            min_value=0,
            max_value=len(df_history) - 1,
            step=1
        )

        if st.button("Hapus Data Terpilih"):
            df_history.drop(index=idx_delete, inplace=True)
            df_history.reset_index(drop=True, inplace=True)
            df_history.to_csv(HISTORY_FILE, index=False)
            st.success("Data berhasil dihapus")
            st.experimental_rerun()

    # Hapus semua data
    with colB:
        if st.button("Hapus Semua Riwayat"):
            os.remove(HISTORY_FILE)
            st.warning("Semua riwayat pasien telah dihapus")
            st.experimental_rerun()

else:
    st.info("Belum ada riwayat pasien.")
