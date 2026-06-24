


import streamlit as st

st.set_page_config(
    page_title="Spotify Churn Prediction",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Churn Prediction")
st.subheader("Artificial Neural Network (ANN) for Customer Churn Prediction")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 📌 Project Overview

    Project ini bertujuan memprediksi kemungkinan pelanggan Spotify berhenti menggunakan layanan (churn)
    menggunakan Artificial Neural Network (ANN).

    Dengan memanfaatkan data perilaku pelanggan, perusahaan dapat mengidentifikasi pelanggan yang berpotensi churn
    lebih awal sehingga strategi retensi dapat dilakukan dengan lebih efektif.
    """)

with col2:
    st.success("""
    🎯 Target Variable

    0 = Tidak Churn

    1 = Churn
    """)

st.markdown("---")

st.subheader("📂 Dataset Features")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    👤 Customer Information

    • Age

    • Gender

    • Country

    • Subscription Type

    • Device Type
    """)

with col2:
    st.info("""
    🎧 User Activity

    • Listening Time

    • Songs Played Per Day

    • Skip Rate

    • Ads Listened Per Week

    • Offline Listening
    """)

st.markdown("---")

st.subheader("🎯 Business Objectives")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
    📉 Reduce Churn

    Mengurangi kehilangan pelanggan.
    """)

with col2:
    st.success("""
    ❤️ Customer Retention

    Meningkatkan loyalitas pelanggan.
    """)

with col3:
    st.success("""
    📊 Data Driven Decision

    Mendukung keputusan bisnis.
    """)

st.markdown("---")

st.subheader("🧠 Model Information")

st.warning("""
Model yang digunakan pada project ini adalah Artificial Neural Network (ANN)
untuk menyelesaikan permasalahan Binary Classification pada prediksi churn pelanggan Spotify.
""")

st.markdown("---")

st.subheader("🚀 Navigasi Aplikasi")

st.write("📊 Dashboard → Menampilkan performa model ANN")

st.write("🔮 Prediksi → Melakukan prediksi churn pelanggan berdasarkan data input")