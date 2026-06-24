


import numpy as np
import pandas as pd
import os
import streamlit as st

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "final_pipeline.pkl")
MODEL_PATH    = os.path.join(BASE_DIR, "spotify_churn_model.keras")

@st.cache_resource(show_spinner="Memuat pipeline...")
def load_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, FunctionTransformer
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.base import BaseEstimator, TransformerMixin
    import joblib
    if not os.path.exists(PIPELINE_PATH):
        st.error(f"File pipeline tidak ditemukan: {PIPELINE_PATH}")
        return None
    try:
        return joblib.load(PIPELINE_PATH)
    except Exception as e:
        st.error(f"Gagal load pipeline: {e}")
        return None

@st.cache_resource(show_spinner="Memuat ANN model...")
def load_ann_model():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

def predict_churn(input_dict: dict) -> dict:
    pipeline = load_pipeline()
    model    = load_ann_model()
    if pipeline is None or model is None:
        return {"label": None, "prob_churn": None, "prob_retain": None}
    df_input    = pd.DataFrame([input_dict])
    X_processed = pipeline.transform(df_input)
    prob_churn  = float(model.predict(X_processed, verbose=0)[0][0])
    label       = int(prob_churn >= 0.5)
    prob_retain = 1.0 - prob_churn
    return {"label": label, "prob_churn": prob_churn, "prob_retain": prob_retain}

FEATURE_COLUMNS      = ["gender", "age", "country", "subscription_type", "listening_time", "songs_played_per_day", "skip_rate", "device_type", "ads_listened_per_week", "offline_listening"]
GENDER_OPTIONS       = ["Male", "Female", "Other"]
COUNTRY_OPTIONS      = ["CA", "DE", "AU", "US", "UK", "IN", "FR", "PK"]
SUBSCRIPTION_OPTIONS = ["Free", "Family", "Premium", "Student"]
DEVICE_OPTIONS       = ["Desktop", "Web", "Mobile"]