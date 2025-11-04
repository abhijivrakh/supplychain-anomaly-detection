# src/realtime_detection.py

import pandas as pd
import joblib
import time
import numpy as np
from feature_engineering import encode_categorical_features, scale_features

from src.utils import load_config

config = load_config()
model_path = config["model"]["model_path"]
scaler_path = config["model"]["scaler_path"]
delay = config["realtime"]["simulation_delay"]


def simulate_realtime_detection(csv_path, delay=0.5):
    """
    Simulates real-time anomaly detection by processing one row at a time.
    """
    # Load trained model and scaler
    model = joblib.load("models/isolation_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Load test data
    df = pd.read_csv(csv_path)

    print("🚀 Starting real-time anomaly simulation...")
    print("=" * 50)

    for i, row in df.head(10).iterrows():

        record = pd.DataFrame([row])  # wrap single row in DataFrame

        # Step 1: Encoding
        encoded = encode_categorical_features(record)

        # Step 2: Align and fill missing columns efficiently
        encoded = encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)


        # Step 3: Scaling
        X_scaled, _, _ = scale_features(encoded)

        # Step 4: Prediction
        prediction = model.predict(X_scaled)[0]
        label = " Normal" if prediction == 1 else " Anomaly"

        print(f"[{i+1}] ➤ Prediction: {label} | Data: {record.iloc[0].to_dict()}")

        time.sleep(delay)  # simulate delay

    print(" Simulation finished.")

if __name__ == "__main__":
    simulate_realtime_detection("data/df_cleaned.csv", delay=0.2)
