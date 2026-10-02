# ❤️ Cardiovascular Disease Prediction

An interactive clinical risk dashboard built with Streamlit, estimating the likelihood of cardiovascular disease from a patient's biometric, blood pressure, metabolic, and lifestyle indicators.

## Overview

Enter a patient's clinical profile — or load one of the built-in presets — and the dashboard returns a cardiovascular risk assessment with an associated risk percentage: either an elevated risk alert or a low risk status. The interface is built as a custom two-panel dashboard: a clinical profile form on the left, and a live architecture/documentation panel on the right.

> **Disclaimer:** this project is a machine learning demonstration built on a public dataset. It is not a medical device and does not provide medical advice or diagnosis.

## Dataset

**[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)** (Kaggle) — 70,000 patient records with a binary cardiovascular disease label.

Features used (11 total):
- **Biometrics:** `age` (converted from days to years), `gender`, `height`, `weight`
- **Blood pressure:** `ap_hi` (systolic), `ap_lo` (diastolic)
- **Metabolic indicators:** `cholesterol`, `gluc` (each on a 3-level scale: normal, above normal, well above normal)
- **Lifestyle:** `smoke`, `alco`, `active`

The raw file is semicolon-delimited; the download script prepares a clean CSV for the app.

## Model & Architecture

- **Preprocessing:** patient age is converted from days to years (`age // 365`), and all 11 features are standardized with `StandardScaler`.
- **Classifier:** a Support Vector Classifier trained on a stratified 80/20 split, with ~73% accuracy under 5-fold cross-validation.
- **Risk score:** the app uses the model's predicted probability when available, and otherwise converts the decision function score to a probability with a sigmoid. A patient is flagged as elevated risk when the model predicts the positive class or the score reaches 50% or above.
- **Pre-compiled artifacts:** the trained model (`svc_model.joblib`) and fitted scaler (`scaler.joblib`) are exported once from the notebook via `joblib` and loaded directly by the app at startup (`@st.cache_resource`) — the model is never retrained at runtime, so assessments are served instantly.
- **Resilient fallback:** if the artifacts aren't found but the dataset is present, the app trains a Logistic Regression classifier on first load and caches the artifacts locally for subsequent runs.
- **Robust path resolution:** artifacts and data files are located by searching multiple candidate directories, so the app runs correctly regardless of where `streamlit run` is invoked from.

## Features

- Preset profiles (high and low cardiovascular risk) for quick testing
- Live BMI calculation from height and weight
- Risk percentage displayed alongside a color-coded assessment card
- Live cross-validation accuracy and feature normalization details
- Custom ruby-themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** — SVC, LogisticRegression, StandardScaler
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/cardiovascular-disease-prediction/download_data.py
```

Ensure `svc_model.joblib` and `scaler.joblib` are present in `projects/cardiovascular-disease-prediction/models/` (exported from the project notebook), or let the app train a model automatically on first run from the downloaded dataset. Then:

```bash
streamlit run Home.py
```

Navigate to the **Cardiovascular Disease Prediction** page from the sidebar.
