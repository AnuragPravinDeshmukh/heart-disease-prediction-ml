"""
Heart Disease Prediction - Preprocessing Module
Handles data loading, feature definitions, categorical encoding, and feature scaling.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart (1).csv")
PKL_DIR = os.path.join(BASE_DIR, "pklfile")
SCALER_PATH = os.path.join(PKL_DIR, "scalar.pkl")
MODEL_PATH = os.path.join(PKL_DIR, "KNN_heart.pkl")
COLUMNS_PATH = os.path.join(PKL_DIR, "columns.pkl")

# Feature classifications
NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
BINARY_COLS = ["FastingBS"]
TARGET_COL = "HeartDisease"

# 15 features expected by the trained model
EXPECTED_COLUMNS = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_M", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST", "ExerciseAngina_Y",
    "ST_Slope_Flat", "ST_Slope_Up"
]

# Baseline standardizer parameters computed from dataset
NUMERICAL_STATS = {
    "Age": {"mean": 53.510893, "std": 9.427478},
    "RestingBP": {"mean": 132.396514, "std": 18.504067},
    "Cholesterol": {"mean": 198.799564, "std": 109.324551},
    "MaxHR": {"mean": 136.809368, "std": 25.446463},
    "Oldpeak": {"mean": 0.887364, "std": 1.065989},
}

def load_dataset(filepath=None):
    """Load the heart disease dataset."""
    path = filepath or DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)

def load_scaler(filepath=None):
    """Load the pre-trained StandardScaler object using joblib."""
    path = filepath or SCALER_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found at: {path}")
    return joblib.load(path)

def load_columns(filepath=None):
    """Load and return the list of expected model feature names."""
    path = filepath or COLUMNS_PATH
    if not os.path.exists(path):
        return EXPECTED_COLUMNS
    try:
        raw_cols = joblib.load(path)
        if callable(raw_cols):
            return list(raw_cols())
        elif hasattr(raw_cols, "__iter__"):
            return list(raw_cols)
    except Exception:
        pass
    return EXPECTED_COLUMNS

def preprocess_patient_input(patient_data, scaler=None):
    """
    Transform raw user input into the exact 15-feature scaled format
    expected by the trained KNN heart disease model.

    Parameters:
    -----------
    patient_data : dict or pd.DataFrame
        Dictionary or 1-row DataFrame with the 11 patient features.
    scaler : StandardScaler, optional
        Pre-loaded StandardScaler instance. If None, it will be loaded.

    Returns:
    --------
    X_scaled : np.ndarray of shape (1, 15)
        Preprocessed and scaled array ready for model inference.
    df_features : pd.DataFrame of shape (1, 15)
        Feature values before second-stage scaling (for debugging/display).
    """
    if scaler is None:
        scaler = load_scaler()

    if isinstance(patient_data, dict):
        df_input = pd.DataFrame([patient_data])
    else:
        df_input = patient_data.copy()

    # Step 1: Standardize numerical columns
    df_norm = pd.DataFrame(index=df_input.index)
    for col in NUMERICAL_COLS:
        val = float(df_input[col].iloc[0])
        stats = NUMERICAL_STATS[col]
        df_norm[col] = [(val - stats["mean"]) / stats["std"]]

    # FastingBS (0 or 1)
    df_norm["FastingBS"] = [int(df_input["FastingBS"].iloc[0])]

    # Step 2: One-hot encode categorical features (drop_first=True alignment)
    # Sex: baseline is 'F', encoded column is 'Sex_M'
    sex_val = str(df_input["Sex"].iloc[0]).strip().upper()
    df_norm["Sex_M"] = [1 if sex_val in ["M", "MALE"] else 0]

    # ChestPainType: baseline is 'ASY', encoded: 'ATA', 'NAP', 'TA'
    cp_val = str(df_input["ChestPainType"].iloc[0]).strip().upper()
    df_norm["ChestPainType_ATA"] = [1 if cp_val == "ATA" else 0]
    df_norm["ChestPainType_NAP"] = [1 if cp_val == "NAP" else 0]
    df_norm["ChestPainType_TA"] = [1 if cp_val == "TA" else 0]

    # RestingECG: baseline is 'LVH', encoded: 'Normal', 'ST'
    ecg_val = str(df_input["RestingECG"].iloc[0]).strip()
    df_norm["RestingECG_Normal"] = [1 if ecg_val == "Normal" else 0]
    df_norm["RestingECG_ST"] = [1 if ecg_val == "ST" else 0]

    # ExerciseAngina: baseline is 'N', encoded: 'ExerciseAngina_Y'
    ea_val = str(df_input["ExerciseAngina"].iloc[0]).strip().upper()
    df_norm["ExerciseAngina_Y"] = [1 if ea_val in ["Y", "YES"] else 0]

    # ST_Slope: baseline is 'Down', encoded: 'Flat', 'Up'
    slope_val = str(df_input["ST_Slope"].iloc[0]).strip()
    df_norm["ST_Slope_Flat"] = [1 if slope_val == "Flat" else 0]
    df_norm["ST_Slope_Up"] = [1 if slope_val == "Up" else 0]

    # Step 3: Ensure exact column order
    df_features = df_norm[EXPECTED_COLUMNS]

    # Step 4: Apply the pre-trained scaler
    X_scaled = scaler.transform(df_features)

    return X_scaled, df_features
