# 👥 Customer Churn Prediction

An interactive subscriber retention dashboard built with Streamlit, estimating the probability that a telecom customer will cancel their service using a Logistic Regression classifier.

## Overview

Enter a customer's account details, subscribed services, and demographics — or load one of the built-in preset profiles — and the dashboard returns a churn probability along with a clear risk assessment: either a high churn risk alert or a likely-to-retain status. The interface is built as a custom two-panel dashboard: a tabbed customer profile form on the left, and a live architecture/documentation panel on the right.

## Dataset

**[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** (Kaggle) — telecom customer records covering demographics, account and contract details, subscribed services, billing information, and a churn label.

Features used (19 total):
- **Account & contract:** `Contract`, `tenure`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, `PaperlessBilling`
- **Subscribed services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`

## Model & Architecture

- **Preprocessing:** each categorical feature is independently label-encoded, blank `TotalCharges` values are treated as zero, and all features are standardized with `StandardScaler`.
- **Classifier:** `LogisticRegression` trained on a stratified 80/20 split, achieving ~80.0% cross-validation accuracy and ~79.8% test accuracy.
- **Decision rule:** a customer is flagged as high churn risk when the predicted churn probability reaches 50% or above.
- **Pre-compiled artifacts:** the trained model (`churn_model.joblib`), fitted scaler (`scaler.joblib`), and per-column encoders (`encoders.joblib`) are exported once from the notebook via `joblib` and loaded directly by the app at startup (`@st.cache_resource` / `@st.cache_data`) — the model is never retrained at runtime, so predictions are served instantly.
- **Resilient fallback:** if the model or scaler artifacts aren't found but the raw dataset is present, the app trains the model on first load and caches the artifacts locally for subsequent runs. If the encoder file is missing, it falls back to deterministic encoders built from the dataset's known category values.
- **Robust path resolution:** artifacts and data files are located by searching multiple candidate directories, so the app runs correctly regardless of where `streamlit run` is invoked from.

## Features

- Tabbed input form: Account & Contract, Subscribed Services, Demographics
- Preset profiles (high churn risk and loyal retained customer) for quick testing
- Churn probability displayed alongside a color-coded risk assessment card
- Live cross-validation and test accuracy metrics
- Custom amber-themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** — LogisticRegression, StandardScaler, LabelEncoder
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/customer-churn-prediction/download_data.py
```

Ensure `churn_model.joblib`, `scaler.joblib`, and `encoders.joblib` are present in `projects/customer-churn-prediction/models/` (exported from the project notebook), or let the app train the model automatically on first run from the downloaded dataset. Then:

```bash
streamlit run Home.py
```

Navigate to the **Customer Churn Prediction** page from the sidebar.
