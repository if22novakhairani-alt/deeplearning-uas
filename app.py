import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.title("Prediksi Penyakit Jantung")

model = load_model("model_jantung_ann.keras")
scaler = joblib.load("scaler.save")

age = st.number_input("Umur", 1, 120)
sex = st.selectbox("Jenis Kelamin", [0, 1])
cp = st.number_input("Chest Pain", 0, 3)
trestbps = st.number_input("Tekanan Darah")
chol = st.number_input("Kolesterol")
fbs = st.selectbox("Gula Darah Tinggi", [0, 1])
restecg = st.number_input("EKG", 0, 2)
thalach = st.number_input("Detak Jantung Maks")
exang = st.selectbox("Angina", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope", 0, 2)

if st.button("Prediksi"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak, slope]])
    data = scaler.transform(data)
    pred = model.predict(data)[0][0]

    if pred > 0.5:
        st.error("Berisiko penyakit jantung")
    else:
        st.success("Tidak berisiko penyakit jantung")
