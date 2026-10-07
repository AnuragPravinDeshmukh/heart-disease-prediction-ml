"""
Model Evaluation View - Metrics, Matrix, Classification Report
"""
import streamlit as st
import pandas as pd
from src.evaluation import compute_model_metrics, plot_confusion_matrix

def render_evaluation_page():
    st.markdown('<div class="main-title">Model Evaluation & Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed diagnostic metrics, confusion matrix, and classification reports</div>', unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Select Machine Learning Model to Inspect:",
        ["KNN (Saved Deployment Model)", "Logistic Regression", "Naive Bayes", "SVM", "Decision Tree"]
    )
    model_key = model_choice.split()[0]
    
    with st.spinner(f"Evaluating {model_choice}..."):
        metrics = compute_model_metrics(model_key)
        
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    with mcol2:
        st.metric("Precision (Class 1)", f"{metrics['precision']:.4f}")
    with mcol3:
        st.metric("Recall (Class 1)", f"{metrics['recall']:.4f}")
    with mcol4:
        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        
    st.markdown("---")
    col_cm, col_cr = st.columns([1, 1])
    with col_cm:
        st.markdown("### 🔲 Confusion Matrix")
        fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], model_choice)
        st.pyplot(fig_cm)
    with col_cr:
        st.markdown("### 📑 Classification Report")
        cr_df = pd.DataFrame(metrics["classification_report"]).transpose().round(4)
        st.dataframe(cr_df, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 📖 Plain-English Metric Definitions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        * **Accuracy:** The overall percentage of correct predictions (both positive and negative) across all test cases.
        * **Precision:** Out of all patients that the model predicted as having heart disease, how many genuinely had it. Critical for minimizing false alarms.
        """)
    with d2:
        st.markdown("""
        * **Recall (Sensitivity):** Out of all patients who genuinely had heart disease, how many were correctly detected by the model. Critical in healthcare to avoid missing positive patients.
        * **F1 Score:** The harmonic mean balancing Precision and Recall. Essential when evaluating balanced clinical utility.
        """)
