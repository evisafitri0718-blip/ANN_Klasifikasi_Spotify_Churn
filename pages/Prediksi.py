


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

st.title("🎯 Prediksi Churn Spotify")

# -------------------------------------------------------
# Form Input
# -------------------------------------------------------
with st.form("form_prediksi"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Profil Dasar**")
        age = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=25, step=1)
        gender = st.selectbox("Jenis Kelamin", GENDER_OPTIONS)
        country = st.selectbox("Negara", COUNTRY_OPTIONS)
        subscription_type = st.selectbox("Tipe Langganan", SUBSCRIPTION_OPTIONS)
        device_type = st.selectbox("Perangkat Utama", DEVICE_OPTIONS)
        premium_since_months = st.number_input("Lama Berlangganan Premium (bulan)", min_value=0, max_value=240, value=12, step=1)

    with col2:
        st.markdown("**Aktivitas & Interaksi**")
        monthly_listening_hours = st.number_input("Jam Mendengarkan / Bulan", min_value=0.0, max_value=744.0, value=30.0, step=0.5)
        num_playlists = st.number_input("Jumlah Playlist Dibuat", min_value=0, max_value=500, value=10, step=1)
        num_songs_liked = st.number_input("Jumlah Lagu Disukai", min_value=0, max_value=5000, value=100, step=1)
        num_artists_followed = st.number_input("Jumlah Artis Diikuti", min_value=0, max_value=500, value=20, step=1)
        num_podcasts_followed = st.number_input("Jumlah Podcast Diikuti", min_value=0, max_value=100, value=5, step=1)
        num_friends = st.number_input("Jumlah Teman", min_value=0, max_value=1000, value=15, step=1)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Pola Penggunaan**")
        shuffle_usage_rate = st.slider("Tingkat Penggunaan Shuffle", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        skip_rate = st.slider("Skip Rate", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        days_since_last_active = st.number_input("Hari Sejak Terakhir Aktif", min_value=0, max_value=365, value=3, step=1)

    with col4:
        st.markdown("**Riwayat Layanan**")
        customer_support_tickets = st.number_input("Tiket Customer Support", min_value=0, max_value=50, value=1, step=1)
        payment_failures = st.number_input("Jumlah Kegagalan Pembayaran", min_value=0, max_value=20, value=0, step=1)
        social_shares = st.number_input("Jumlah Social Shares", min_value=0, max_value=500, value=5, step=1)

    submitted = st.form_submit_button("🔍 Prediksi Churn", use_container_width=True, type="primary")

# -------------------------------------------------------
# Proses Prediksi
# -------------------------------------------------------
if submitted:
    input_dict = {
        "age"                     : age,
        "gender"                  : gender,
        "country"                 : country,
        "subscription_type"       : subscription_type,
        "device_type"             : device_type,
        "monthly_listening_hours" : monthly_listening_hours,
        "num_playlists"           : num_playlists,
        "num_songs_liked"         : num_songs_liked,
        "num_artists_followed"    : num_artists_followed,
        "num_podcasts_followed"   : num_podcasts_followed,
        "days_since_last_active"  : days_since_last_active,
        "num_friends"             : num_friends,
        "shuffle_usage_rate"      : shuffle_usage_rate,
        "skip_rate"               : skip_rate,
        "premium_since_months"    : premium_since_months,
        "customer_support_tickets": customer_support_tickets,
        "payment_failures"        : payment_failures,
        "social_shares"           : social_shares,
    }

    with st.spinner("Memproses prediksi..."):
        result = predict_churn(input_dict)

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if result["label"] is None:
        st.error("❌ Prediksi gagal. Pastikan file model dan pipeline tersedia.")
    else:
        prob_churn  = result["prob_churn"]
        prob_retain = result["prob_retain"]

        if result["label"] == 1:
            st.error(f"⚠️ **CHURN** — Pelanggan berpotensi berhenti berlangganan")
        else:
            st.success(f"✅ **TIDAK CHURN** — Pelanggan kemungkinan tetap berlangganan")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Probabilitas Churn", f"{prob_churn:.2%}")
        with col_b:
            st.metric("Probabilitas Tidak Churn", f"{prob_retain:.2%}")

        st.progress(prob_churn)
        st.caption("Threshold klasifikasi: 0.50")

st.caption("Prediksi · ANN Project Spotify Churn")