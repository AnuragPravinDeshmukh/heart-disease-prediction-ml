# ❤️ Heart Disease Prediction

### End-to-End Machine Learning • Streamlit • Python

> An interactive **Heart Disease Prediction system** built with Machine Learning and Streamlit, featuring data analysis, model comparison, evaluation, and real-time predictions.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</p>

---

## 🚀 What This Project Does

The application takes clinical health parameters as input and uses a trained **K-Nearest Neighbors (KNN)** model to predict whether a patient is likely to have heart disease.

**Workflow:**

```text
📊 Dataset
    ↓
🧹 Data Preprocessing
    ↓
🔎 Exploratory Data Analysis
    ↓
⚙️ Feature Encoding & Scaling
    ↓
🤖 Model Training & Comparison
    ↓
📈 Evaluation
    ↓
❤️ KNN Prediction
    ↓
🌐 Streamlit Web App
```

---

## 📊 Model Performance

| Model                  |   Accuracy |   F1 Score |
| ---------------------- | ---------: | ---------: |
| 🥇 Logistic Regression | **87.13%** | **0.8870** |
| SVM                    |     86.14% |     0.8800 |
| Naive Bayes            |     85.81% |     0.8746 |
| KNN                    |     84.49% |     0.8630 |
| Decision Tree          |     75.91% |     0.7795 |

The deployed application uses **KNN with a pre-trained StandardScaler**.

---

## 🧠 Tech Stack

**Python** • **Pandas** • **NumPy** • **Scikit-learn** • **Matplotlib** • **Seaborn** • **Streamlit**

---

## ✨ Features

* ❤️ Interactive heart disease prediction
* 📊 Exploratory Data Analysis
* 🤖 Multiple ML model comparison
* 📈 Accuracy & F1-score evaluation
* 🔲 Confusion matrix
* 📁 Interactive dataset viewer
* 📖 Data dictionary
* ⚙️ Pre-trained model & scaler
* 🌐 Streamlit web interface

---

## 📂 Project Structure

```text
heart-disease-prediction-ml/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── heart (1).csv
│
├── pklfile/
│   ├── KNN_heart.pkl
│   ├── scalar.pkl
│   └── columns.pkl
│
└── src/
    ├── preprocessing.py
    ├── prediction.py
    ├── evaluation.py
    └── views/
        ├── home_view.py
        ├── prediction_view.py
        ├── comparison_view.py
        ├── evaluation_view.py
        ├── eda_view.py
        ├── dataset_view.py
        └── about_view.py
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction-ml.git

cd heart-disease-prediction-ml

pip install -r requirements.txt

streamlit run app.py
```

The application will open in your browser.

---

## 📌 Dataset

The project uses a heart disease dataset containing **918 records and 12 original features**.

The application performs preprocessing and feature transformation before passing the data to the trained model.

---

## ⚠️ Disclaimer

This project is created for **educational and demonstration purposes only**.

It is **not a medical diagnostic system** and should not be used for real-world medical decisions.

---

## 👨‍💻 Author

**Anurag Deshmukh**

B.Tech — Artificial Intelligence

📌 Data Science • Machine Learning • Python • SQL

---

⭐ **If you found this project useful, consider giving it a star!**
