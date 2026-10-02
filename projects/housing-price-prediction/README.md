# 🏠 Delhi Housing Price Predictor

An interactive real estate valuation dashboard that estimates property prices across Delhi using a pre-trained Linear Regression model, served instantly via Streamlit.

## Overview

Select a location, enter the property area, choose bedroom count and resale status, and toggle available amenities — the dashboard returns an instant price estimate, automatically formatted in Lakhs or Crores depending on magnitude. The interface is built as a custom two-panel dashboard: an interactive input form on the left, and a live architecture/documentation panel on the right.

## Dataset

**[Housing Prices in Metropolitan Areas of India](https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india)** (Kaggle) — `Delhi.csv`, containing property listings across Delhi with 40 features spanning location, size, bedroom count, resale status, and amenities (gymnasium, swimming pool, security, parking, and more).

## Model & Architecture

- **Algorithm:** Linear Regression, trained on standardized numerical features and label-encoded location data.
- **Preprocessing:** `LabelEncoder` for `Location`, `StandardScaler` applied across all input features, with the target variable scaled to millions during training for numerical stability and converted back to full Rupee values at inference.
- **Pre-compiled artifacts:** the trained model (`housing_price_model.joblib`) and fitted scaler (`housing_scaler.joblib`) are exported once from the notebook via `joblib` and loaded directly by the app at startup (`@st.cache_resource`) — the model is never retrained at runtime, so predictions are served instantly rather than waiting on a fresh training pass on every load.
- **Graceful fallback:** if the raw dataset isn't present locally, the app falls back to a bundled default list of Delhi localities so the location selector still functions.

## Features

- Location selector covering major Delhi localities
- Area, bedroom count, and resale status inputs
- Expandable amenities panel covering 30+ property features
- Automatic Lakhs/Crores formatting based on predicted value
- Custom-themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** — LinearRegression, StandardScaler, LabelEncoder
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/housing-price-prediction/download_data.py
```

Ensure `housing_price_model.joblib` and `housing_scaler.joblib` are present in `projects/housing-price-prediction/models/` (exported from the project notebook). Then:

```bash
streamlit run Home.py
```

Navigate to the **Delhi Housing Price Predictor** page from the sidebar.
