# 🎵 Spotify Song Recommendation System

An interactive audio-based song recommendation dashboard built with Streamlit, matching tracks by acoustic similarity using cosine distance across standardized audio features.

## Overview

Search for a song by title — or pick from a curated top-10 list — and the system returns a ranked list of acoustically similar tracks, each shown with artist, album, and genre context. The interface is built as a custom two-panel dashboard: an interactive search panel on the left, and a live architecture/documentation panel on the right.

## Dataset

**[30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)** (Kaggle) — ~32,800 tracks with playlist metadata and audio features.

Features used:
- **Categorical:** `playlist_genre`, `playlist_subgenre`, `playlist_name`, `track_artist`, `track_album_name`
- **Numerical:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

## Model & Architecture

- **Preprocessing:** each categorical feature is independently label-encoded; all numerical audio features are standardized via `StandardScaler`.
- **Similarity:** a precomputed cosine similarity matrix is calculated once across the full combined feature set, capturing acoustic closeness between every track pair.
- **Pre-compiled artifacts:** the processed track catalog (`spotify_df.joblib`), similarity matrix (`spotify_similarity.joblib`), scaler (`spotify_scaler.joblib`), and encoders (`spotify_encoders.joblib`) are exported once from the notebook via `joblib` and loaded directly by the app at startup — no retraining or refitting occurs at runtime, so recommendations are served instantly.
- **Flexible matching:** queries first attempt an exact, case-insensitive title match, then fall back to a partial substring match if no exact match is found — making the search tolerant of minor variations in how a title is typed.
- **Graceful fallback:** if the pre-compiled artifacts haven't been generated yet, the app falls back to a small built-in sample track library so the interface remains functional and demonstrable without the full dataset.

## Features

- Free-text search with partial-match tolerance, or quick-select from top 10 tracks
- Adjustable recommendation count (1–20 results)
- Each result displayed with artist, album, and genre tags
- Live dataset metrics (tracks in archive, similarity engine used)
- Custom Spotify-inspired themed UI with a dedicated architecture/documentation panel

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** — LabelEncoder, StandardScaler, cosine_similarity
- **Streamlit** — dashboard interface and caching
- **joblib** — model artifact serialization

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/spotify-song-recommendation-system/download_data.py
```

Ensure `spotify_df.joblib`, `spotify_similarity.joblib`, `spotify_scaler.joblib`, and `spotify_encoders.joblib` are present in `projects/spotify-song-recommendation-system/model/` (exported from the project notebook). Then:

```bash
streamlit run Home.py
```

> **Note:** `spotify_similarity.joblib` (the precomputed similarity matrix, ~8 GB) is not included in this repository because it exceeds GitHub's file size limits. The other artifacts are included. If the similarity matrix is missing, the app rebuilds it from the dataset on first load and reuses it afterward. This first-run build takes a while and needs a large amount of RAM. Alternatively, run the project notebook to export the file yourself.

Navigate to the **Spotify Song Recommendation System** page from the sidebar.
