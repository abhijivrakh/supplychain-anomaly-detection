from feature_engineering import preprocess_pipeline

# Run the pipeline and print results to verify
try:
    X, scaler, feature_names = preprocess_pipeline(save_scaler=True)
    print("Feature engineering ran successfully.")
    print(f" Shape of final features: {X.shape}")
    print(f" First 5 features: {feature_names[:5]}")
except Exception as e:
    print(" Error occurred:", e)
