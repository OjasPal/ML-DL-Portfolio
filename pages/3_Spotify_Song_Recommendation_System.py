import difflib
import os
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------- Page config ----------
st.set_page_config(page_title="Spotify Song Recommender", page_icon="🎵", layout="centered")


# ---------- Data loading + preprocessing (cached so it only runs once) ----------
@st.cache_data(show_spinner="Loading and preprocessing dataset...")
def load_data():
    data_path = "projects/spotify-song-recommendation-system/data/spotify_songs.csv"
    if not os.path.exists(data_path):
        alt_path = "data/spotify_songs.csv"
        if os.path.exists(alt_path):
            data_path = alt_path

    df = pd.read_csv(data_path)
    df["index"] = df.index
    df["track_name"] = df["track_name"].astype(str)
    print(df["track_name"].apply(type).value_counts())

    cat_columns = [
        "playlist_genre",
        "playlist_subgenre",
        "playlist_name",
        "track_artist",
        "track_album_name",
    ]
    num_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    for col in cat_columns:
        df[col] = df[col].fillna("")
    for col in num_columns:
        df[col] = df[col].fillna(0)

    # Preprocessing identical to notebook
    df_processed = df.copy()
    encoders = {}
    for col in cat_columns:
        encoders[col] = LabelEncoder()
        df_processed[col] = encoders[col].fit_transform(df_processed[col].astype(str))

    scaler = StandardScaler()
    df_processed[num_columns] = scaler.fit_transform(df_processed[num_columns])

    df_features = df_processed[cat_columns + num_columns]
    return df, df_features


@st.cache_resource(show_spinner="Building similarity model... (this happens once)")
def build_similarity_matrix(df_features):
    similarity = cosine_similarity(df_features)
    return similarity


def get_recommendations(track_name, df, similarity, top_n=10):
    list_of_all_songs = [str(x) for x in df["track_name"].tolist()]
    find_close_match = difflib.get_close_matches(track_name, list_of_all_songs)

    if not find_close_match:
        return None, []

    close_match = find_close_match[0]
    index_of_the_song = df[df.track_name == close_match]["index"].values[0]

    similarity_score = list(enumerate(similarity[index_of_the_song]))
    sorted_similar_songs = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for song in sorted_similar_songs[1: top_n + 1]:  # skip index 0 = the song itself
        idx = song[0]
        title = df[df.index == idx]["track_name"].values[0]
        artist = df[df.index == idx]["track_artist"].values[0]
        recommendations.append(f"{title} — {artist}")

    return close_match, recommendations


# ---------- UI ----------
st.title("🎵 Spotify Song Recommendation System")
st.write("Type a song you like, and get similar song suggestions based on audio features and playlists.")

df, df_features = load_data()
similarity = build_similarity_matrix(df_features)

track_name = st.text_input("Enter your favourite spotify song:", placeholder="e.g. Blinding Lights")

top_n = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)

if st.button("Get Recommendations", type="primary"):
    if not track_name.strip():
        st.warning("Please enter a song name first.")
    else:
        close_match, recommendations = get_recommendations(track_name, df, similarity, top_n=top_n)

        if close_match is None:
            st.error("Couldn't find a close match for that song. Try checking the spelling.")
        else:
            st.success(f"Showing results for: **{close_match}**")
            for i, title in enumerate(recommendations, start=1):
                st.write(f"{i}. {title}")

st.divider()
st.caption("Built with Streamlit • Data: 30,000 Spotify Songs Dataset (Kaggle)")