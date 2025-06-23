# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_clean_data(path="data/df_cleaned.csv"):
    """
    Load the cleaned dataset from the data folder.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    return df


def encode_categorical_features(df):
    """
    Encode categorical features using One-Hot Encoding (if any).
    """
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded


def scale_features(df_encoded, save_scaler=False, scaler_path="models/scaler.pkl"):
    """
    Scale numerical features using StandardScaler.
    
    Parameters:
        df_encoded: pandas DataFrame with only numerical columns.
        save_scaler: if True, saves the fitted scaler for deployment.
        scaler_path: path to save the scaler if required.

    Returns:
        scaled_features: Numpy array of scaled data
        scaler: fitted StandardScaler object
        feature_names: list of column names used
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_encoded)
    feature_names = df_encoded.columns.tolist()

    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return scaled_features, scaler, feature_names


def preprocess_pipeline(path="data/df_cleaned.csv", save_scaler=False):
    """
    Complete preprocessing pipeline:
        - Load cleaned dataset
        - Encode categorical features
        - Scale features

    Returns:
        scaled_features, scaler, feature_names
    """
    df = load_clean_data(path)
    df_encoded = encode_categorical_features(df)
    scaled_features, scaler, feature_names = scale_features(
        df_encoded, save_scaler=save_scaler
    )
    return scaled_features, scaler, feature_names
