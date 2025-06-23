# src/anomaly_model.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import os

from feature_engineering import preprocess_pipeline

def train_anomaly_model(X, save_model=True, model_path="models/isolation_forest.pkl"):
    """
    Train Isolation Forest on preprocessed data and optionally save it.
    """
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)

    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")

    return model


def generate_anomaly_labels(model, X):
    """
    Predict anomalies: 1 = normal, -1 = anomaly
    """
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    print(" Starting anomaly detection pipeline...")

    # Step 1: Preprocess features
    X, scaler, features = preprocess_pipeline(save_scaler=False)

    # Step 2: Train Isolation Forest
    model = train_anomaly_model(X)

    # Step 3: Generate anomaly labels
    predictions = generate_anomaly_labels(model, X)

    # Step 4: Output anomaly counts
    import numpy as np
    unique, counts = np.unique(predictions, return_counts=True)
    results = dict(zip(unique, counts))
    print("📊 Anomaly Label Distribution:", results)

    print("✅ Anomaly detection completed.")
