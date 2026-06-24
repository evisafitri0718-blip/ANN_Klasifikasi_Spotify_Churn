


import os
import sys
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.predict import (
    predict_churn,
    GENDER_OPTIONS,
    COUNTRY_OPTIONS,
    SUBSCRIPTION_OPTIONS,
    DEVICE_OPTIONS,
)

st.title("Prediksi Churn Spotify")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        country = st.selectbox("Country", COUNTRY_OPTIONS)
        subscription_type = st.selectbox("Subscription Type", SUBSCRIPTION_OPTIONS)
        device_type = st.selectbox("Device Type", DEVICE_OPTIONS)

    with col2:
        listening_time = st.number_input("Listening Time", min_value=0, value=100)
        songs_played_per_day = st.number_input("Songs Played Per Day", min_value=0, value=20)
        skip_rate = st.number_input("Skip Rate", min_value=0.0, max_value=1.0, value=0.2)
        ads_listened_per_week = st.number_input("Ads Listened Per Week", min_value=0, value=10)
        offline_listening = st.number_input("Offline Listening", min_value=0, value=1)

    submitted = st.form_submit_button("Prediksi", use_container_width=True, type="primary")

if submitted:
    input_dict = {
        "gender"              : gender,
        "age"                 : age,
        "country"             : country,
        "subscription_type"   : subscription_type,
        "listening_time"      : listening_time,
        "songs_played_per_day": songs_played_per_day,
        "skip_rate"           : skip_rate,
        "device_type"         : device_type,
        "ads_listened_per_week": ads_listened_per_week,
        "offline_listening"   : offline_listening,
    }

    with st.spinner("Memproses..."):
        result = predict_churn(input_dict)

    st.divider()
    st.subheader("Hasil Prediksi")

    if result["label"] is None:
        st.error("Prediksi gagal.")
    else:
        prob_churn  = result["prob_churn"]
        prob_retain = result["prob_retain"]

        if result["label"] == 1:
            st.error("CHURN - Pelanggan berpotensi berhenti berlangganan")
        else:
            st.success("TIDAK CHURN - Pelanggan kemungkinan tetap berlangganan")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Probabilitas Churn", f"{prob_churn:.2%}")
        with col_b:
            st.metric("Probabilitas Tidak Churn", f"{prob_retain:.2%}")

        st.progress(prob_churn)