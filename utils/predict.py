

"""
utils/predict.py
----------------
Fungsi utilitas untuk memuat model, pipeline, dan menjalankan prediksi.
"""

import numpy as np
import pandas as pd
import os
import sys
import streamlit as st

# -------------------------------------------------------
# Konstanta path
# -------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "final_pipeline.pkl")
MODEL_PATH    = os.path.join(BASE_DIR, "spotify_churn_model.keras")


# -------------------------------------------------------
# Load Pipeline dengan sklearn imports lengkap
# -------------------------------------------------------
@st.cache_resource(show_spinner="Memuat pipeline...")
def load_pipeline():
    """Memuat preprocessing pipeline dari file .pkl"""

    # ✅ Import SEMUA komponen sklearn yang mungkin ada di pipeline
    # (harus di-import sebelum joblib.load agar pickle bisa resolve class-nya)
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        OneHotEncoder, OrdinalEncoder, LabelEncoder,
        FunctionTransformer
    )
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.base import BaseEstimator, TransformerMixin

    import joblib

    if not os.path.exists(PIPELINE_PATH):
        st.error(f"❌ File pipeline tidak ditemukan: {PIPELINE_PATH}")
        return None

    try:
        pipeline = joblib.load(PIPELINE_PATH)
        return pipeline
    except AttributeError as e:
        st.error(
            f"❌ Gagal load pipeline: {e}\n\n"
            "Kemungkinan pipeline dibuat dengan custom class atau versi scikit-learn berbeda. "
            "Pastikan versi di requirements.txt sama dengan saat training."
        )
        return None


# -------------------------------------------------------
# Load Model ANN
# -------------------------------------------------------
@st.cache_resource(show_spinner="Memuat ANN model...")
def load_ann_model():
    """Memuat model ANN Keras dari file .keras"""
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File model tidak ditemukan: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        return None


# -------------------------------------------------------
# Fungsi prediksi utama
# -------------------------------------------------------
def predict_churn(input_dict: dict) -> dict:
    """
    Melakukan prediksi churn berdasarkan input dictionary.

    Parameters
    ----------
    input_dict : dict
        Dictionary berisi nilai setiap fitur pengguna.

    Returns
    -------
    dict dengan keys:
        - 'label'       : int (0 = Tidak Churn, 1 = Churn)
        - 'prob_churn'  : float (probabilitas churn 0-1)
        - 'prob_retain' : float (probabilitas tidak churn 0-1)
    """
    pipeline = load_pipeline()
    model    = load_ann_model()

    if pipeline is None or model is None:
        return {"label": None, "prob_churn": None, "prob_retain": None}

    # Buat DataFrame dari input
    df_input = pd.DataFrame([input_dict])

    # Preprocessing
    X_processed = pipeline.transform(df_input)

    # Prediksi
    prob_churn  = float(model.predict(X_processed, verbose=0)[0][0])
    label       = int(prob_churn >= 0.5)
    prob_retain = 1.0 - prob_churn

    return {
        "label"      : label,
        "prob_churn" : prob_churn,
        "prob_retain": prob_retain,
    }


# -------------------------------------------------------
# Nama kolom yang digunakan pipeline
# -------------------------------------------------------
FEATURE_COLUMNS = [
    "age",
    "gender",
    "country",
    "subscription_type",
    "device_type",
    "monthly_listening_hours",
    "num_playlists",
    "num_songs_liked",
    "num_artists_followed",
    "num_podcasts_followed",
    "days_since_last_active",
    "num_friends",
    "shuffle_usage_rate",
    "skip_rate",
    "premium_since_months",
    "customer_support_tickets",
    "payment_failures",
    "social_shares",
]

# Nilai kategori valid (sesuai dataset training)
GENDER_OPTIONS       = ["Male", "Female", "Other"]
COUNTRY_OPTIONS      = ["USA", "UK", "Germany", "France", "Australia",
                         "Canada", "India", "Brazil", "Japan", "Other"]
SUBSCRIPTION_OPTIONS = ["Free", "Premium", "Family", "Student"]
DEVICE_OPTIONS       = ["Mobile", "Desktop", "Tablet", "Smart TV"]

