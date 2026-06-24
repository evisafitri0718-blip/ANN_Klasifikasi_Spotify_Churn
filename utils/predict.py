


import joblib
from tensorflow.keras.models import load_model

print("ANN FILE BERHASIL DIBUKA")

# Load Pipeline
print("Loading Pipeline...")
pipeline = joblib.load("final_pipeline.pkl")
print("Pipeline Loaded")

# Load Model ANN
print("Loading Model...")
model = load_model("spotify_churn_model.keras")
print("Model Loaded")


def predict_churn(data):

    # Transform data menggunakan pipeline
    data_transform = pipeline.transform(data)

    # Prediksi ANN
    prediction = model.predict(data_transform, verbose=0)

    # Tampilkan hasil mentah ANN di terminal
    print("=" * 50)
    print("HASIL RAW ANN:")
    print(prediction)
    print("=" * 50)

    # Ambil probabilitas
    probability = float(prediction[0][0])

    # Tentukan hasil klasifikasi
    if probability >= 0.5:
        result = "Churn"
    else:
        result = "Tidak Churn"

    return result, probability