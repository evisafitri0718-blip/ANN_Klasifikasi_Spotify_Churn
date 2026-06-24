


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================
# DASHBOARD
# ====================================

st.title("📊 Dashboard Model ANN Spotify Churn")

# ====================================
# HASIL MODEL ANN
# ====================================

accuracy = 0.75
precision = 0.22
recall = 0.01
f1 = 0.01

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "75%")

with col2:
    st.metric("Precision", "22%")

with col3:
    st.metric("Recall", "1%")

with col4:
    st.metric("F1 Score", "1%")

st.divider()

# ====================================
# GRAFIK PERFORMA MODEL
# ====================================

st.subheader("📈 Performa Model ANN")

metric_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
})

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    metric_df["Metric"],
    metric_df["Value"]
)

ax.set_ylim(0,1)
ax.set_ylabel("Score")
ax.set_title("Performa Model ANN")

st.pyplot(fig)

st.success("""
💡 Insight Model

• Accuracy menunjukkan kemampuan model dalam mengklasifikasikan pelanggan.

• Precision masih rendah sehingga prediksi churn belum optimal.

• Recall sangat rendah sehingga masih banyak pelanggan churn yang tidak terdeteksi.

• Model ANN masih dapat ditingkatkan melalui balancing data dan hyperparameter tuning.
""")

st.divider()

# ====================================
# BUSINESS INSIGHT
# ====================================

st.subheader("📌 Business Insight")

st.info("""
Model ANN digunakan untuk memprediksi kemungkinan pelanggan Spotify melakukan churn.

Manfaat bagi perusahaan:

• Mengidentifikasi pelanggan yang berpotensi churn lebih awal.

• Menentukan target program retensi pelanggan.

• Mengurangi kehilangan pelanggan.

• Meningkatkan loyalitas pelanggan.

• Mendukung pengambilan keputusan berbasis data.
""")

# ====================================
# VISUALISASI BISNIS
# ====================================

st.subheader("🎵 Subscription Type vs Churn")

data_subscription = pd.DataFrame({
    "Subscription": ["Free","Family","Premium","Student"],
    "Churn": [510,530,535,515],
    "Tidak Churn": [1520,1385,1590,1450]
})

fig, ax = plt.subplots(figsize=(8,5))

data_subscription.set_index("Subscription").plot(
    kind="bar",
    ax=ax
)

ax.set_title("Subscription Type vs Churn")
ax.set_ylabel("Jumlah Pelanggan")

st.pyplot(fig)

st.info("""
💡 Insight

• Paket Premium memiliki jumlah pelanggan aktif tertinggi.

• Paket Family menunjukkan jumlah churn yang relatif tinggi.

• Strategi retensi dapat difokuskan pada pelanggan Free dan Family.
""")

st.divider()

# ====================================
# ENGAGEMENT SCORE
# ====================================

st.subheader("📊 Distribusi Engagement Score")

engagement = pd.DataFrame({
    "Score":[1000,3000,5000,7000,10000,15000,20000,25000,30000],
    "Jumlah":[900,700,500,400,300,200,120,60,10]
})

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    engagement["Score"],
    engagement["Jumlah"],
    marker="o"
)

ax.set_title("Distribusi Engagement Score")
ax.set_xlabel("Engagement Score")
ax.set_ylabel("Jumlah Pelanggan")

st.pyplot(fig)

st.info("""
💡 Insight

• Mayoritas pelanggan memiliki engagement score rendah hingga menengah.

• Semakin tinggi engagement score maka jumlah pelanggan semakin sedikit.

• Pelanggan dengan engagement rendah berpotensi lebih mudah churn.
""")

st.divider()

# ====================================
# PROJECT SUMMARY
# ====================================

col1, col2 = st.columns(2)

with col1:
    st.success("""
🎯 Tujuan Project

Memprediksi kemungkinan pelanggan Spotify melakukan churn menggunakan Artificial Neural Network (ANN).
""")

with col2:
    st.warning("""
🤖 Model

Artificial Neural Network (ANN)

Binary Classification:

• 0 = Tidak Churn

• 1 = Churn
""")