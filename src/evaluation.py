"""
Heart Disease Prediction - Model Evaluation & Comparison Module
Provides evaluation metrics, confusion matrices, classification reports,
and comparison charts for the tested machine learning models.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import (
    load_dataset,
    load_scaler,
    NUMERICAL_COLS,
    NUMERICAL_STATS,
    EXPECTED_COLUMNS
)
from src.prediction import load_model

# Benchmark metrics recorded from the existing ML project
BENCHMARK_RESULTS = {
    "Logistic Regression": {
        "Accuracy": 0.8713,
        "F1 Score": 0.8870,
        "Precision": 0.8621,
        "Recall": 0.9136,
        "Description": "Linear classifier using logistic sigmoid function with L2 regularization."
    },
    "SVM": {
        "Accuracy": 0.8614,
        "F1 Score": 0.8800,
        "Precision": 0.8485,
        "Recall": 0.9136,
        "Description": "Support Vector Classifier with Radial Basis Function (RBF) kernel."
    },
    "Naive Bayes": {
        "Accuracy": 0.8581,
        "F1 Score": 0.8746,
        "Precision": 0.8537,
        "Recall": 0.8969,
        "Description": "Probabilistic classifier applying Bayes' theorem with feature independence assumption."
    },
    "KNN": {
        "Accuracy": 0.8449,
        "F1 Score": 0.8630,
        "Precision": 0.8415,
        "Recall": 0.8859,
        "Description": "Instance-based learner classifying by majority vote among 5 nearest neighbors."
    },
    "Decision Tree": {
        "Accuracy": 0.7591,
        "F1 Score": 0.7795,
        "Precision": 0.7656,
        "Recall": 0.7935,
        "Description": "Non-parametric tree partition model splitting on information gain / Gini impurity."
    }
}

def get_comparison_dataframe():
    """Return a styled pandas DataFrame of the model comparison benchmark."""
    data = []
    for model_name, metrics in BENCHMARK_RESULTS.items():
        data.append({
            "Model": model_name,
            "Accuracy": f"{metrics['Accuracy'] * 100:.2f}%",
            "F1 Score": f"{metrics['F1 Score']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}",
            "Accuracy_Raw": metrics["Accuracy"],
            "F1_Raw": metrics["F1 Score"],
            "Description": metrics["Description"]
        })
    df_comp = pd.DataFrame(data)
    return df_comp.sort_values(by="Accuracy_Raw", ascending=False).reset_index(drop=True)

def get_evaluation_data():
    """
    Prepare standard train/test split matching the project benchmark.
    Returns X_train_s, X_test_s, y_train, y_test.
    """
    df = load_dataset()
    df_copy = df.copy()
    
    # Standardize numerical features
    for col in NUMERICAL_COLS:
        stats = NUMERICAL_STATS[col]
        df_copy[col] = (df_copy[col] - stats["mean"]) / stats["std"]
        
    X_enc = pd.get_dummies(df_copy.drop("HeartDisease", axis=1), drop_first=True)
    X_enc = X_enc[EXPECTED_COLUMNS]
    y = df["HeartDisease"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.33, random_state=120
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    return X_train_s, X_test_s, y_train, y_test

def get_trained_model(model_name):
    """Retrieve or train model instance for evaluation."""
    X_train_s, X_test_s, y_train, y_test = get_evaluation_data()
    
    if model_name == "KNN":
        try:
            return load_model(), X_test_s, y_test
        except Exception:
            pass
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    else:
        model = KNeighborsClassifier(n_neighbors=5)
        
    model.fit(X_train_s, y_train)
    return model, X_test_s, y_test

def compute_model_metrics(model_name):
    """
    Compute full evaluation metrics for the selected model.
    """
    model, X_test_s, y_test = get_trained_model(model_name)
    y_pred = model.predict(X_test_s)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": cr,
        "y_test": y_test,
        "y_pred": y_pred
    }

def plot_confusion_matrix(cm, model_name="Model"):
    """Generate a clean seaborn confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Heart Disease (0)", "Heart Disease (1)"],
        yticklabels=["No Heart Disease (0)", "Heart Disease (1)"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    return fig

def plot_model_comparison_chart(metric="Accuracy"):
    """Generate a clean horizontal bar comparison chart."""
    df_comp = get_comparison_dataframe()
    fig, ax = plt.subplots(figsize=(8, 4))
    
    col = "Accuracy_Raw" if metric == "Accuracy" else "F1_Raw"
    df_sorted = df_comp.sort_values(by=col, ascending=True)
    
    colors = ["#2563eb" if m == "Logistic Regression" else "#93c5fd" for m in df_sorted["Model"]]
    bars = ax.barh(df_sorted["Model"], df_sorted[col], color=colors, height=0.55)
    
    for bar in bars:
        w = bar.get_width()
        txt = f"{w * 100:.2f}%" if metric == "Accuracy" else f"{w:.4f}"
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, txt,
                ha="left", va="center", fontsize=10, weight="bold", color="#1e293b")
        
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Model Comparison - {metric}", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel(f"{metric} Score", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
