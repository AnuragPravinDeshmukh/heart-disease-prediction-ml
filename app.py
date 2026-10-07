"""
Heart Disease Prediction - Professional Streamlit ML Application
Main entry point and multi-page coordinator.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Enable running directly with `python app.py` or VS Code Run button
if __name__ == "__main__":
    if not st.runtime.exists():
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())

from src.views.home_view import render_home_page
from src.views.prediction_view import render_prediction_page
from src.views.comparison_view import render_comparison_page
from src.views.evaluation_view import render_evaluation_page
from src.views.eda_view import render_eda_page
from src.views.dataset_view import render_dataset_page
from src.views.about_view import render_about_page

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction ML",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .custom-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-card {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .step-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.85rem;
        text-align: center;
        font-weight: 600;
        color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("## ❤️ Heart Disease ML")
    st.markdown("*Clinical Classification Platform*")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "❤️ Prediction",
            "🤖 Model Comparison",
            "📊 Model Evaluation",
            "📈 EDA",
            "📁 Dataset",
            "ℹ️ About"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Model Information")
    st.info("**Primary Model:** K-Nearest Neighbors (KNN)\n\n**Feature Count:** 15 Scaled Features\n\n**Training Dataset:** 918 records")
    st.caption("Educational Machine Learning Project — Not for Medical Diagnosis.")

# Page Router
if selected_page == "🏠 Home":
    render_home_page()
elif selected_page == "❤️ Prediction":
    render_prediction_page()
elif selected_page == "🤖 Model Comparison":
    render_comparison_page()
elif selected_page == "📊 Model Evaluation":
    render_evaluation_page()
elif selected_page == "📈 EDA":
    render_eda_page()
elif selected_page == "📁 Dataset":
    render_dataset_page()
elif selected_page == "ℹ️ About":
    render_about_page()
