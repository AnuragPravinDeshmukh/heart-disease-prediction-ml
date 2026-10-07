"""
About View - Methodology, Architecture, and Project Background
"""
import streamlit as st

def render_about_page():
    st.markdown('<div class="main-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Architecture, machine learning methodology, and project documentation</div>', unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        ### 🎯 Problem Statement
        Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year.
        The objective of this project is to develop and deploy an accessible, reliable, and interpretable machine learning 
        classification application to assist in predicting whether a patient exhibits heart disease risk based on routine 
        clinical, physiological, and stress-test indicators.
        
        ### 🧠 Machine Learning Formulation
        * **Task Type:** Supervised Binary Classification
        * **Target Variable:** `HeartDisease` (0 = No Disease, 1 = Heart Disease)
        * **Dataset Size:** 918 observations, 11 features
        * **Evaluation Strategy:** Train-Test Split (33% hold-out test set)
        """)
    with col_p2:
        st.markdown("""
        ### 🛠️ Technology Stack
        * **Frontend & Web Framework:** [Streamlit](https://streamlit.io/)
        * **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
        * **Machine Learning Library:** [Scikit-learn](https://scikit-learn.org/)
        * **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
        * **Model Persistence:** [Joblib](https://joblib.readthedocs.io/)
        
        ### 🤖 Tested Algorithms
        1. **Logistic Regression** (Highest Accuracy: 87.13%, F1: 0.8870)
        2. **Support Vector Machine (SVM)** (86.14% Accuracy)
        3. **Naive Bayes** (85.81% Accuracy)
        4. **K-Nearest Neighbors (KNN)** (84.49% Accuracy - Serialized model)
        5. **Decision Tree Classifier** (75.91% Accuracy)
        """)
        
    st.markdown("---")
    st.markdown("### 📋 End-to-End ML Pipeline Architecture")
    st.markdown("""
    ```
    Raw Patient Input (11 Features)
          ↓
    Standardize 5 Continuous Features (Age, BP, Chol, MaxHR, Oldpeak)
          ↓
    One-Hot Encode Categorical Features (drop_first=True alignment)
          ↓
    Align into 15-Feature Vector (Expected Model Columns)
          ↓
    Pre-trained StandardScaler (pklfile/scalar.pkl)
          ↓
    Pre-trained Model Inference (pklfile/KNN_heart.pkl)
          ↓
    Output Prediction Class + Prediction Probability Breakdown
    ```
    """)
    
    st.markdown("""
    <div class="disclaimer-card">
        <h4 style="color:#b91c1c; margin-top:0;">⚠️ Educational & Clinical Disclaimer</h4>
        <p style="color:#7f1d1d; margin-bottom:0;">
        This tool is created for educational, research, and portfolio demonstration purposes. It does not 
        constitute medical diagnostic advice. Always consult certified medical practitioners for healthcare decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
