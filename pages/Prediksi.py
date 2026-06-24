

import streamlit as st
import pandas as pd

from utils.predict import predict_churn

st.title("🎯 Prediksi Churn Spotify")

# Input Numerik
age = st.number_input("Age", min_value=10, max_value=100, value=25)

listening_time = st.number_input(
    "Listening Time",
    min_value=0,
    value=100
)

songs_played_per_day = st.number_input(
    "Songs Played Per Day",
    min_value=0,
    value=20
)

skip_rate = st.number_input(
    "Skip Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.2
)

ads_listened_per_week = st.number_input(
    "Ads Listened Per Week",
    min_value=0,
    value=10
)

offline_listening = st.number_input(
    "Offline Listening",
    min_value=0,
    value=1
)

# Input Kategorikal
gender = st.selectbox(
    "Gender",
    ['Female', 'Other', 'Male']
)

country = st.selectbox(
    "Country",
    ['CA', 'DE', 'AU', 'US', 'UK', 'IN', 'FR', 'PK']
)

subscription_type = st.selectbox(
    "Subscription Type",
    ['Free', 'Family', 'Premium', 'Student']
)

device_type = st.selectbox(
    "Device Type",
    ['Desktop', 'Web', 'Mobile']
)

# Tombol Prediksi
if st.button("Predict"):

    input_data = pd.DataFrame({
        'age': [age],
        'listening_time': [listening_time],
        'songs_played_per_day': [songs_played_per_day],
        'skip_rate': [skip_rate],
        'ads_listened_per_week': [ads_listened_per_week],
        'offline_listening': [offline_listening],
        'gender': [gender],
        'country': [country],
        'subscription_type': [subscription_type],
        'device_type': [device_type]
    })

    result, probability = predict_churn(input_data)

    st.subheader("Hasil Prediksi")

    if result == "Churn":
        st.error(f"⚠️ Pelanggan Berpotensi Churn ({probability:.2%})")
    else:
        st.success(f"✅ Pelanggan Tidak Churn ({probability:.2%})")

    st.progress(float(probability))