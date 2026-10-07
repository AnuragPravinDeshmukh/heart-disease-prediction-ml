"""
Model Comparison View - Comparative Benchmarks
"""
import streamlit as st
from src.evaluation import get_comparison_dataframe, plot_model_comparison_chart

def render_comparison_page():
    st.markdown('<div class="main-title">Model Comparison & Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparative evaluation across 5 supervised machine learning algorithms</div>', unsafe_allow_html=True)
    
    st.markdown("""
    During the machine learning experimental phase, five supervised classification algorithms were trained and evaluated
    using the heart disease dataset under standardized test conditions.
    """)
    
    df_comp = get_comparison_dataframe()
    st.dataframe(
        df_comp[["Model", "Accuracy", "F1 Score", "Precision", "Recall", "Description"]],
        use_container_width=True,
        hide_index=True
    )
    
    st.info("Based on the current evaluation results, **Logistic Regression** achieved the highest Accuracy (87.13%) and F1 Score (0.8870) among the tested models, followed closely by Support Vector Machine (SVM) and Naive Bayes.")
    
    st.markdown("---")
    st.markdown("### 📊 Performance Comparison Charts")
    chart_metric = st.radio("Select Metric to Visualize:", ["Accuracy", "F1 Score"], horizontal=True)
    chart_fig = plot_model_comparison_chart(metric=chart_metric)
    st.pyplot(chart_fig)
    
    st.markdown("---")
    st.markdown("### 🔬 Algorithmic Observations")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        #### Linear & Kernel Classifiers:
        * **Logistic Regression:** Demonstrates robust linear separability across standardized continuous variables and one-hot encoded clinical factors.
        * **Support Vector Machine (SVM):** The RBF kernel captures high-dimensional boundaries effectively, yielding an 86.14% accuracy and 0.8800 F1 score.
        """)
    with col_b:
        st.markdown("""
        #### Probabilistic & Instance-Based Classifiers:
        * **Naive Bayes:** Shows strong diagnostic capability (85.81% accuracy) despite the feature independence assumption.
        * **K-Nearest Neighbors (KNN):** Selected for saved deployment (84.49% accuracy, 0.8630 F1 score), providing intuitive distance-based neighborhood inference and class probability estimation.
        * **Decision Tree:** Lower generalization (75.91% accuracy) due to orthogonal axis-aligned splitting and susceptibility to local variance.
        """)
