# 🎵 Spotify Song Recommendation System

A content-based song recommender that suggests similar tracks based on audio features and playlist metadata — deployed as an interactive Streamlit page within the [ML-DL-Portfolio](../../) app.

## Overview

Type in a song you like, and the app finds the closest matching track in the dataset, then returns a ranked list of the most similar songs — based on audio characteristics (danceability, energy, tempo, etc.) and playlist/genre context, not listening history.

This project also includes genre segmentation via K-Means clustering in the original notebook, grouping songs into natural clusters based on their audio profile, independent of the labeled genre — useful for exploring how songs group together beyond the dataset's own genre tags.

## Dataset

**[30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)** (Kaggle) — ~32,800 tracks with playlist metadata and detailed audio features.

Columns used:
- **Categorical:** `playlist_genre`, `playlist_subgenre`, `playlist_name`, `track_artist`, `track_album_name` — label-encoded
- **Numerical:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo` — standardized via `StandardScaler`
- `track_name` — used for matching user input and displaying recommendations

## Model

- **Approach:** Content-based filtering using a mix of categorical and numerical audio features (not collaborative filtering — no listening/rating history involved)
- **Preprocessing:** each categorical column gets its own independently fitted `LabelEncoder`; all numerical features are standardized together
- **Similarity:** cosine similarity computed across all songs' combined feature vectors
- **Segmentation (notebook only):** K-Means clustering with the elbow method used to determine an optimal number of clusters, grouping songs by audio profile similarity

## How It Works

1. User types a song title (typos/partial names are fine)
2. `difflib.get_close_matches()` finds the closest matching title in the dataset
3. The matched song's row in the similarity matrix is retrieved and sorted by score, highest first
4. Top N most similar songs (excluding the song itself), each shown with its artist, are displayed — N adjustable via a slider

## Tech Stack

- **Python**, **Pandas**
- **scikit-learn** — LabelEncoder, StandardScaler, cosine_similarity, KMeans
- **Streamlit** — interactive UI

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/spotify-song-recommendation-system/download_data.py

# Run the full multi-page app
streamlit run Home.py
```

Then navigate to the **Spotify Song Recommendation System** page from the sidebar.

## Notes

Originally built as "Spotify Songs Genre Segmentation," combining unsupervised clustering with a content-based recommender. The deployed app focuses on the recommendation feature; the clustering/genre segmentation analysis remains in the original notebook for reference.
