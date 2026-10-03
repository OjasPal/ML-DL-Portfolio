# 🫀 Heart Disease Prediction

An interactive clinical screening dashboard built with Streamlit, estimating the likelihood of heart disease from 13 diagnostic biomarkers using a standardized Logistic Regression model.

## Overview

Enter a patient's demographics, exercise stress test results, and blood chemistry findings — or load one of the built-in clinical presets — and the dashboard returns a prediction with an associated probability, shown in a color-coded result card. The interface is built as a custom two-panel dashboard: a tabbed diagnostic form on the left, and a live architecture/documentation panel on the right.

> **Disclaimer:** this project is a machine learning demonstration built on a public dataset. It is not a medical device and does not provide medical advice or diagnosis.

## Dataset

**[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)** (Kaggle) — patient records with 13 clinical features and a binary target label indicating heart disease status as labeled in the source dataset.

Features used (13 total), grouped as they appear in the dashboard:
- **Demographics & vitals:** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar)
- **Exercise & ECG stress:** `restecg` (resting ECG result), `thalach` (maximum heart rate achieved), `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope` (peak exercise ST segment slope)
- **Blood chemistry & pathology:** `ca` (major vessels colored by fluoroscopy), `thal` (thalassemia blood flow result)

## Model & Architecture

- **Preprocessing:** all 13 features are standardized with `StandardScaler`, using the exact scaling parameters learned during training.
- **Classifier:** `LogisticRegression`, trained on a stratified 80/20 split.
- **Decision rule:** a patient is flagged when the model predicts the positive class or the predicted probability reaches 50% or above; the probability is shown alongside the result.
- **Pre-compiled artifacts:** the trained model (`heart_disease_model.joblib`) and fitted scaler (`heart_scaler.joblib`) are exported once from the notebook via `joblib` and loaded directly by the app at startup (`@st.cache_resource`) — the model is never retrained at runtime, so predictions are served instantly.
- **Resilient fallback:** if the artifacts aren't found but the raw dataset is present, the app trains the model on first load and caches the artifacts locally for subsequent runs.
- **Robust path resolution:** artifacts and data files are located by searching multiple candidate directories, so the app runs correctly regardless of where `streamlit run` is invoked from.

## Features

- Tabbed input form: Demographics & Vitals, Exercise & ECG Stress, Blood Chemistry & Pathology
- Clinical preset profiles for quick testing
- Probability displayed alongside a color-coded result card
- Custom crimson-themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** — LogisticRegression, StandardScaler
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/heart-disease-prediction/download_data.py
```

Ensure `heart_disease_model.joblib` and `heart_scaler.joblib` are present in `projects/heart-disease-prediction/models/` (exported from the project notebook), or let the app train the model automatically on first run from the downloaded dataset. Then:

```bash
streamlit run Home.py
```

Navigate to the **Heart Disease Prediction** page from the sidebar.
