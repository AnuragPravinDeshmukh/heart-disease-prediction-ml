"""
EDA View - Exploratory Data Analysis & Clinical Visualizations
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_dataset

def render_eda_page():
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Clinical patterns, correlations, and distribution insights from 918 patients</div>', unsafe_allow_html=True)
    
    df = load_dataset()
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["🎯 Target & Demographics", "🩺 Vital Signs & Lab Metrics", "⚡ Exercise & ECG Factors"])
    
    with eda_tab1:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### Heart Disease Target Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = df["HeartDisease"].value_counts()
            bars = ax.bar(["0: No Heart Disease", "1: Heart Disease"], counts, color=["#10b981", "#ef4444"], width=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 8, f"{h} ({h/len(df)*100:.1f}%)", ha="center", weight="bold")
            ax.set_ylim(0, max(counts) * 1.15)
            ax.set_ylabel("Patient Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_t2:
            st.markdown("#### Heart Disease by Biological Sex")
            fig, ax = plt.subplots(figsize=(6, 4))
            sex_target = pd.crosstab(df["Sex"], df["HeartDisease"])
            sex_target.plot(kind="bar", stacked=False, color=["#10b981", "#ef4444"], ax=ax, width=0.6)
            ax.set_xticklabels(["Female (F)", "Male (M)"], rotation=0)
            ax.set_ylabel("Patient Count")
            ax.set_xlabel("Sex")
            ax.legend(["No Heart Disease (0)", "Heart Disease (1)"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("#### Age Distribution by Heart Disease Status")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        sns.histplot(data=df, x="Age", hue="HeartDisease", kde=True, palette=["#10b981", "#ef4444"], bins=20, ax=ax)
        ax.set_title("Age Distribution (Green = Healthy, Red = Heart Disease)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        
    with eda_tab2:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### Serum Cholesterol Distribution (mg/dl)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x="HeartDisease", y="Cholesterol", palette=["#10b981", "#ef4444"], ax=ax, width=0.4)
            ax.set_xticklabels(["No Heart Disease (0)", "Heart Disease (1)"])
            ax.set_ylabel("Cholesterol (mg/dl)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_v2:
            st.markdown("#### Resting Blood Pressure (mm Hg)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x="HeartDisease", y="RestingBP", palette=["#10b981", "#ef4444"], ax=ax, width=0.4)
            ax.set_xticklabels(["No Heart Disease (0)", "Heart Disease (1)"])
            ax.set_ylabel("RestingBP (mm Hg)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("#### Maximum Heart Rate Achieved (MaxHR)")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        sns.kdeplot(data=df, x="MaxHR", hue="HeartDisease", fill=True, palette=["#10b981", "#ef4444"], ax=ax)
        ax.set_title("Max Heart Rate Density (Healthy patients reach higher peak heart rates)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with eda_tab3:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("#### Heart Disease by Chest Pain Type")
            fig, ax = plt.subplots(figsize=(6, 4))
            cp_target = pd.crosstab(df["ChestPainType"], df["HeartDisease"])
            cp_target.plot(kind="bar", color=["#10b981", "#ef4444"], ax=ax, width=0.6)
            ax.set_ylabel("Patient Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(["No Disease", "Heart Disease"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_e2:
            st.markdown("#### Exercise-Induced Angina vs Disease")
            fig, ax = plt.subplots(figsize=(6, 4))
            ea_target = pd.crosstab(df["ExerciseAngina"], df["HeartDisease"])
            ea_target.plot(kind="bar", color=["#10b981", "#ef4444"], ax=ax, width=0.5)
            ax.set_xticklabels(["No (N)", "Yes (Y)"], rotation=0)
            ax.set_ylabel("Patient Count")
            ax.legend(["No Disease", "Heart Disease"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("#### Peak Exercise ST Slope vs Heart Disease")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        st_target = pd.crosstab(df["ST_Slope"], df["HeartDisease"])
        st_target.plot(kind="bar", color=["#10b981", "#ef4444"], ax=ax, width=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel("Patient Count")
        ax.set_title("Flat/Down slopes correlate strongly with positive HeartDisease cases")
        ax.legend(["No Disease", "Heart Disease"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
