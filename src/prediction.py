"""
Heart Disease Prediction - Prediction Module
Handles loading the trained machine learning model and executing inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import (
    MODEL_PATH,
    SCALER_PATH,
    load_scaler,
    preprocess_patient_input
)

def load_model(filepath=None):
    """
    Load the pre-trained KNN Heart Disease classification model.
    """
    path = filepath or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)

def predict_heart_disease(patient_data, model=None, scaler=None):
    """
    Execute end-to-end heart disease prediction for a patient profile.

    Parameters:
    -----------
    patient_data : dict or pd.DataFrame
        Patient features matching the dataset specification.
    model : estimator, optional
        Pre-loaded model. If None, it will be loaded from disk.
    scaler : StandardScaler, optional
        Pre-loaded scaler. If None, it will be loaded from disk.

    Returns:
    --------
    result : dict
        Dictionary containing prediction class, human-readable label,
        probabilities, confidence score, and feature details.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    # Apply preprocessing pipeline
    X_scaled, df_features = preprocess_patient_input(patient_data, scaler=scaler)

    # Generate prediction
    prediction_raw = model.predict(X_scaled)[0]
    prediction = int(prediction_raw)

    # Check for probability support
    prob_0 = None
    prob_1 = None
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_scaled)[0]
            prob_0 = float(probabilities[0])
            prob_1 = float(probabilities[1])
            confidence = prob_1 if prediction == 1 else prob_0
        except Exception:
            pass

    # Risk level categorization based purely on predicted probability
    if prob_1 is not None:
        if prob_1 < 0.35:
            risk_level = "Low"
            risk_color = "#10b981"  # green
        elif prob_1 < 0.65:
            risk_level = "Moderate"
            risk_color = "#f59e0b"  # amber
        else:
            risk_level = "High"
            risk_color = "#ef4444"  # red
    else:
        risk_level = "High" if prediction == 1 else "Low"
        risk_color = "#ef4444" if prediction == 1 else "#10b981"

    label = "Class 1 — Heart Disease" if prediction == 1 else "Class 0 — No Heart Disease"

    return {
        "prediction": prediction,
        "label": label,
        "probability_0": prob_0,
        "probability_1": prob_1,
        "confidence": confidence,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "features": df_features,
        "scaled_vector": X_scaled
    }
