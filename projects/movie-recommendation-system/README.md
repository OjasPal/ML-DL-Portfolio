# 🎬 Movie Recommendation System

A content-based movie recommender that suggests similar movies based on genre, plot, keywords, and tagline — deployed as an interactive Streamlit page within the [ML-DL-Portfolio](../../) app.

## Overview

Type in a movie you like, and the app finds the closest matching title in the dataset, then returns a ranked list of the most similar movies — based purely on textual content (no user ratings or viewing history required).

## Dataset

**[Full TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)** (Kaggle) — trimmed down to the top 50,000 movies by vote count, to keep the similarity computation fast while still covering a broad, relevant range of films.

Columns used:
- `genres`, `keywords`, `overview`, `tagline` — combined into a single text field that captures each movie's content and theme
- `title` — used for matching user input and displaying recommendations

## Model

- **Approach:** Content-based filtering (not collaborative filtering — no user rating data is used)
- **Vectorization:** `TfidfVectorizer` (scikit-learn) converts each movie's combined text into a numerical vector, weighting distinctive words more heavily than common ones
- **Similarity:** Cosine similarity computed across all movie vectors, producing a similarity score between every pair of movies

## How It Works

1. User types a movie title (typos/partial names are fine)
2. `difflib.get_close_matches()` finds the closest matching title in the dataset
3. The matched movie's row in the similarity matrix is retrieved and sorted by score, highest first
4. Top N most similar movies (excluding the movie itself) are displayed, with N adjustable via a slider

## Tech Stack

- **Python**, **Pandas**
- **scikit-learn** — TfidfVectorizer, cosine_similarity
- **Streamlit** — interactive UI

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt
python projects/movie-recommendation-system/download_data.py

# Run the full multi-page app
streamlit run Home.py
```

Then navigate to the **Movie Recommendation System** page from the sidebar.

## Notes

This was the first project in this portfolio — originally built as a 1st-year beginner project and revisited two years later with an updated, larger dataset and a Streamlit interface. The core recommendation logic (TF-IDF + cosine similarity) is unchanged from the original notebook; only the dataset size and the UI layer were added.
