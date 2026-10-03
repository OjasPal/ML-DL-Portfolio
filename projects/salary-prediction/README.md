# 💰 Software Salary Prediction

An interactive compensation estimator built with Streamlit, predicting annual salaries (in Lakhs per Annum) for software professionals in India using a CatBoost gradient boosting model with domain-specific feature engineering.

## Overview

Enter a company, job title, location, employment status, and company rating — or load one of the built-in industry presets — and the dashboard returns an estimated annual salary in LPA and rupees, an 80% prediction interval, and a comparison against typical market tiers. Before predicting, it shows the engineered features it extracted from your input (company tier, seniority, tech domain, and job level). The interface is built as a custom two-panel dashboard: a profile form on the left, and a live architecture/documentation panel on the right.

## Dataset

**[Software Professional Salaries 2022](https://www.kaggle.com/datasets/iamsouravbanerjee/software-professional-salaries-2022)** (Kaggle) — self-reported salary records for software professionals in India, using the `Salary_Dataset_with_Extra_Features.csv` file. The model is trained on 22,687 records after outlier filtering (salaries under 0.5 LPA or above 100 LPA are removed to prevent distortion).

## Model & Architecture

- **Algorithm:** `CatBoostRegressor` with native categorical feature handling.
- **Target transformation:** salaries are modeled on a `log1p` scale to handle the right-skewed distribution, and predictions are converted back to LPA with `expm1`.
- **Feature engineering:**
  - **Company normalization:** regex-based alias resolution (e.g. "TCS" → "Tata Consultancy Services"), suffix stripping, and a junk-entry filter
  - **Company tiering:** seven tiers — Top Tech, Product Startup, Finance, Large Corporation, Service IT, Other, and Unknown
  - **Seniority and level parsing:** intern, junior, mid-level, senior/lead, and architect/principal levels, plus SDE I/II/III style level numbers extracted from job titles
  - **Domain clustering:** job titles grouped into tech domains such as Data/AI, DevOps/Cloud, Backend, Frontend, Mobile, Fullstack, and QA
  - **Location features:** city alias normalization, major tech hub, and Bangalore indicators
- **Prediction intervals:** an 80% interval is returned with every estimate, using separate error quantiles for companies seen versus unseen during training.
- **Pre-compiled artifact:** the trained model, feature order, company frequency table, known-company list, and interval quantiles are bundled into a single artifact (`salary_catboost_artifact.joblib`), exported once from the notebook via `joblib` and loaded at startup (`@st.cache_resource`) — nothing is retrained at runtime.
- **Fallback estimator:** if the artifact can't be loaded, the app falls back to a rule-based estimator calibrated to approximate the model's learned relationships, keeping the interface functional.

## Performance

Evaluated on held-out data:

| Metric | Value |
|---|---|
| R² (log scale) | 0.356 |
| Median percentage error | 37.8% |
| Predictions within ±25% | 33.7% |
| Mean absolute error | 3.19 LPA |

The dataset is self-reported and compensation depends on many factors it doesn't capture (skills, negotiation, company-specific pay bands), so estimates are best read as a market range rather than a precise figure — which is why every prediction ships with an 80% interval.

## Features

- Free-text company and job title input with live engineered-feature preview
- Industry benchmark presets (senior ML engineer, SDE II backend, service IT, and intern profiles)
- Estimated salary in LPA and annual rupees, with an 80% interval and monthly breakdown
- Market tier comparison chart
- Custom golden-themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **CatBoost** — gradient boosting regressor
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/salary-prediction/download_data.py
```

Ensure `salary_catboost_artifact.joblib` is present in `projects/salary-prediction/models/` (exported from the project notebook). Then:

```bash
streamlit run Home.py
```

Navigate to the **Software Salary Prediction** page from the sidebar.
