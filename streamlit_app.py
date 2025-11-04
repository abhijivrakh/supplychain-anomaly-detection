# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

from src.feature_engineering import encode_categorical_features, scale_features

# Load saved model and scaler
model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("📦 Real-Time Anomaly Detection for Supply Chain")
st.markdown("Upload your CSV file and detect operational anomalies.")

# Upload file
uploaded_file = st.file_uploader("Upload supply chain data (.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        # Step 1: Encode
        df_encoded = encode_categorical_features(df)

        # Step 2: Ensure feature alignment
        model_features = scaler.feature_names_in_
        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Add missing columns with 0

        df_encoded = df_encoded[model_features]  # Align column order

        # Step 3: Scale
        X_scaled, _, _ = scale_features(df_encoded)

        # Step 4: Predict
        predictions = model.predict(X_scaled)
        df["Anomaly"] = np.where(predictions == -1, " Anomaly", "Normal")

        # Display
        st.subheader("🔍 Prediction Summary")
        st.write(df[["Anomaly"]].value_counts().rename("Count").reset_index())

        st.subheader("📋 Full Output with Anomaly Labels")
        st.dataframe(df)

        # Download result
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Result CSV", csv, file_name="anomaly_results.csv")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
