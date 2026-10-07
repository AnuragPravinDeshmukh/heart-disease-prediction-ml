"""
Dataset View - Clinical Data Exploration & Dictionary
"""
import streamlit as st
import pandas as pd
from src.preprocessing import load_dataset

def render_dataset_page():
    st.markdown('<div class="main-title">Clinical Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed structural overview of the Heart Disease Dataset</div>', unsafe_allow_html=True)
    
    df = load_dataset()
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Rows", f"{df.shape[0]}")
    with col_m2:
        st.metric("Total Columns", f"{df.shape[1]}")
    with col_m3:
        st.metric("Missing Values", f"{df.isnull().sum().sum()}")
    with col_m4:
        st.metric("Duplicate Rows", f"{df.duplicated().sum()}")
        
    st.markdown("---")
    st.markdown("### 📋 Interactive Data Viewer")
    st.dataframe(df, use_container_width=True, height=350)
    
    st.markdown("---")
    st.markdown("### 📊 Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📖 Feature Data Dictionary")
    data_dict = [
        {"Feature": "Age", "Type": "Numeric (int)", "Description": "Age of the patient in years", "Range / Values": "28 - 77"},
        {"Feature": "Sex", "Type": "Categorical", "Description": "Biological sex of the patient", "Range / Values": "M (Male), F (Female)"},
        {"Feature": "ChestPainType", "Type": "Categorical", "Description": "Chest pain classification", "Range / Values": "TA (Typical), ATA (Atypical), NAP (Non-Anginal), ASY (Asymptomatic)"},
        {"Feature": "RestingBP", "Type": "Numeric (int)", "Description": "Resting blood pressure in mm Hg", "Range / Values": "80 - 200 mm Hg"},
        {"Feature": "Cholesterol", "Type": "Numeric (int)", "Description": "Serum cholesterol in mg/dl", "Range / Values": "0 - 603 mg/dl"},
        {"Feature": "FastingBS", "Type": "Binary (int)", "Description": "Fasting blood sugar > 120 mg/dl", "Range / Values": "0 (Normal / <=120), 1 (Elevated / >120)"},
        {"Feature": "RestingECG", "Type": "Categorical", "Description": "Resting electrocardiographic results", "Range / Values": "Normal, ST (Abnormality), LVH (Ventricular Hypertrophy)"},
        {"Feature": "MaxHR", "Type": "Numeric (int)", "Description": "Maximum heart rate achieved during test", "Range / Values": "60 - 202 bpm"},
        {"Feature": "ExerciseAngina", "Type": "Categorical", "Description": "Exercise-induced angina symptom", "Range / Values": "N (No), Y (Yes)"},
        {"Feature": "Oldpeak", "Type": "Numeric (float)", "Description": "ST depression induced by exercise vs rest", "Range / Values": "-2.6 to 6.2 mm"},
        {"Feature": "ST_Slope", "Type": "Categorical", "Description": "Peak exercise ST segment slope", "Range / Values": "Up (Upsloping), Flat, Down (Downsloping)"},
        {"Feature": "HeartDisease", "Type": "Binary (int) [TARGET]", "Description": "Clinical classification outcome", "Range / Values": "0: No Heart Disease, 1: Heart Disease"}
    ]
    st.dataframe(pd.DataFrame(data_dict), use_container_width=True, hide_index=True)
