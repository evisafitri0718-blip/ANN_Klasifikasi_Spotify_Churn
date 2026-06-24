from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import os
from tensorflow.keras.models import load_model

print("ANN FILE BERHASIL DIBUKA")

# ✅ Gunakan path absolut berdasarkan lokasi file ini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PIPELINE_PATH = os.path.join(BASE_DIR, "final_pipeline.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "spotify_churn_model.keras")

# ✅ Import semua yang dipakai saat membuat pipeline di notebook
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load Pipeline
print(f"Loading Pipeline dari: {PIPELINE_PATH}")
print(f"File exists: {os.path.exists(PIPELINE_PATH)}")  # debug
pipeline = joblib.load(PIPELINE_PATH)
print("Pipeline Loaded ✅")

# Load Model ANN
print(f"Loading Model dari: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")  # debug
model = load_model(MODEL_PATH)
print("Model Loaded ✅")


def predict_churn(data):
    # Transform data menggunakan pipeline
    data_transform = pipeline.transform(data)

    # Prediksi ANN
    prediction = model.predict(data_transform, verbose=0)

    print("=" * 50)
    print("HASIL RAW ANN:")
    print(prediction)
    print("=" * 50)

    probability = float(prediction[0][0])
    result = "Churn" if probability >= 0.5 else "Tidak Churn"

    return result, probability

