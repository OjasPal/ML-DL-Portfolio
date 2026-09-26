# 🏠 Delhi Housing Price Prediction

A machine learning model that estimates apartment and house prices in Delhi based on area, location, number of bedrooms, resale status, and available amenities — deployed as an interactive Streamlit page within the [ML-DL-Portfolio](../../) app.

## Overview

This project uses historical real estate data to train a Linear Regression model that predicts property prices in Indian Rupees. Users can select a location, enter the property area, choose the number of bedrooms, and toggle available amenities (swimming pool, gym, security, parking, etc.) to get an instant price estimate.

## Dataset

**[Housing Prices in Metropolitan Areas of India](https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india)** (Kaggle) — specifically `Delhi.csv`, containing 4,998 property listings across 40 features.

Key columns used:
- `Price` — target variable (scaled to millions during training for numerical stability)
- `Area` — property size in square feet
- `Location` — locality within Delhi (label-encoded)
- `No. of Bedrooms`, `Resale` — core property attributes
- 35+ binary amenity columns (Gymnasium, SwimmingPool, ClubHouse, 24X7Security, CarParking, etc.)

**Note:** the dataset's creator used `9` to mark amenities with no available information (not necessarily absent in real life) — this is expected and not a data quality issue.

## Model

- **Algorithm:** Linear Regression (scikit-learn)
- **Preprocessing:** `LabelEncoder` for the `Location` column, `StandardScaler` applied to all features
- **Split:** `StratifiedShuffleSplit` (80/20) stratified on the `Resale` column to keep new/resale property ratios consistent across train and test sets
- **Evaluation metric:** RMSE (Root Mean Squared Error) on the held-out test set

The app trains the model fresh on startup (cached via `st.cache_resource` so it only runs once per session) rather than loading a pre-trained file — keeping the deployed app self-contained and always in sync with the current dataset.

## How It Works

1. User selects location, area, bedroom count, resale status, and amenities via the sidebar/form
2. Inputs are encoded and scaled using the same transformations applied during training
3. The trained model predicts a price (in millions), which is converted back to actual Rupees
4. Result is displayed in ₹, with an automatic Lakhs/Crores conversion for readability

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** — LinearRegression, StandardScaler, LabelEncoder, StratifiedShuffleSplit
- **Streamlit** — interactive UI

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/housing-price-prediction/download_data.py

# Run the full multi-page app
streamlit run Home.py
```

Then navigate to the **Delhi Housing Price Predictor** page from the sidebar.

## Notes

This was originally built as a beginner project and later adapted into this portfolio with a Streamlit interface. The modeling approach (single Linear Regression, no hyperparameter tuning or feature engineering beyond encoding/scaling) is intentionally kept simple to reflect the original learning exercise — the focus of this update was making it usable and presentable, not maximizing predictive accuracy.
