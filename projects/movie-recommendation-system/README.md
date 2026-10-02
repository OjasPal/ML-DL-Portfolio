# 🎬 Movie Recommendation System

An interactive content-based movie recommendation dashboard built with Streamlit, using TF-IDF vectorization and cosine similarity to surface movies similar to any title a user enters.

## Overview

Enter a movie title — or pick from a curated preset list — and the system returns a ranked list of the most similar titles, computed from each movie's genres, keywords, overview, and tagline. The interface is built as a custom two-panel dashboard: an interactive query panel on the left, and a live documentation/architecture panel on the right detailing how the pipeline works under the hood.

## Dataset

**[TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)** (Kaggle), trimmed to the top 50,000 movies by vote count for a balance of coverage and performance.

Features used: `genres`, `keywords`, `overview`, `tagline` — combined into a single text representation per movie.

## Model & Architecture

- **Vectorization:** `TfidfVectorizer` transforms each movie's combined text into a weighted feature vector.
- **Similarity:** rather than precomputing and storing a full N×N similarity matrix (which scales quadratically and becomes memory-prohibitive at this dataset size), similarity is computed **on-the-fly** per query — only the selected movie's vector is compared against the full feature matrix at request time. This keeps memory usage flat regardless of dataset size while still returning results instantly.
- **Pre-compiled artifacts:** the processed dataset (`movies_data.pkl`) and TF-IDF feature vectors (`feature_vectors.pkl`) are exported once from the notebook via `joblib` and loaded directly by the app at startup (`@st.cache_resource`) — the vectorizer is never refit at runtime, so the app starts instantly rather than reprocessing 50,000 rows on every load.
- **Fuzzy matching:** `difflib.get_close_matches` resolves minor typos or partial titles to the closest actual entry in the dataset.

## Features

- Preset quick-select for popular titles, or free-text search for any movie
- Adjustable recommendation count (5–30 results)
- Live dataset metrics (total movies indexed, similarity algorithm used)
- Custom dark-themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**
- **scikit-learn** — TfidfVectorizer, cosine_similarity
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/movie-recommendation-system/download_data.py
```

Ensure `movies_data.pkl` and `feature_vectors.pkl` are present in `projects/movie-recommendation-system/models/` (exported from the project notebook). Then:

```bash
streamlit run Home.py
```

Navigate to the **Movie Recommendation System** page from the sidebar.
