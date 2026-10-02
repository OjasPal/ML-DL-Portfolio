# 📧 Email Spam Detector

An interactive email classification dashboard built with Streamlit, using TF-IDF feature extraction and Logistic Regression to distinguish spam from legitimate (ham) messages.

## Overview

Paste or type the subject and body of an email — or select from a set of built-in spam/ham sample presets — and the dashboard classifies it instantly, returning both a prediction and a confidence score. The interface is built as a custom two-panel dashboard: an interactive classification panel on the left, and a live architecture/documentation panel on the right.

## Dataset

**[Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)** (Kaggle) — a labeled collection of emails.

Columns used:
- `text` — email subject and body
- `label_num` — binary target (0 = ham, 1 = spam); the `label` column holds the same classes as ham/spam strings

## Model & Architecture

- **Text preprocessing:** each email is cleaned via regex (stripping non-alphabetic characters), lowercased, stopword-filtered, and stemmed using NLTK's `PorterStemmer` — identical to the original notebook pipeline.
- **Vectorization:** `TfidfVectorizer` transforms cleaned text into weighted feature vectors.
- **Classifier:** `LogisticRegression`, achieving ~99.4% training accuracy and ~98.2% testing accuracy on an 80/20 stratified split.
- **Pre-compiled artifacts:** the trained model (`spam_model.pkl`) and fitted vectorizer (`tfidf_vectorizer.pkl`) are exported once from the notebook via `joblib` and loaded directly by the app at startup — classification runs instantly rather than retraining on every load.
- **Resilient fallback:** if pre-compiled artifacts aren't found, the app automatically retrains from the raw CSV on the fly, with dynamic column detection to handle variations in column naming (`text`/`Message`/`email`, `spam`/`label`/`Category`) — keeping the dashboard functional even before artifacts are generated.
- **Robust path resolution:** model, vectorizer, and dataset files are located by searching multiple candidate directories, so the app runs correctly regardless of where `streamlit run` is invoked from.

## Features

- Built-in spam and ham sample presets alongside free-text input
- Confidence score displayed alongside each prediction
- Live training/testing accuracy metrics
- Custom electric-blue themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **NLTK** — stopword filtering, Porter stemming
- **scikit-learn** — TfidfVectorizer, LogisticRegression
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/email-spam-detector/download_data.py
```

Ensure `spam_model.pkl` and `tfidf_vectorizer.pkl` are present in `projects/email-spam-detector/models/` (exported from the project notebook), or let the app train them automatically on first run from `data/emails.csv`. Then:

```bash
streamlit run Home.py
```

Navigate to the **Email Spam Detector** page from the sidebar.
