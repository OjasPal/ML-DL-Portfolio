
import difflib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page config ----------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


# ---------- Data loading + model building (cached so it only runs once) ----------
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    movies = pd.read_csv("projects/movie-recommendation-system/data/tmdb_movies_50k.csv")
    movies["index"] = movies.index

    selected_features = ["genres", "keywords", "overview", "tagline"]
    for feature in selected_features:
        movies[feature] = movies[feature].fillna("")

    combined_features = (
        movies["genres"] + " " + movies["keywords"] + " " + movies["overview"] + " " + movies["tagline"]
    )
    return movies, combined_features


@st.cache_resource(show_spinner="Building similarity model... (this happens once)")
def build_similarity_matrix(combined_features):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return similarity


def get_recommendations(movie_name, movies, similarity, top_n=30):
    list_of_all_titles = movies["title"].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return None, []

    close_match = find_close_match[0]
    index_of_the_movie = movies[movies.title == close_match]["index"].values[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie in sorted_similar_movies[1: top_n + 1]:  # skip index 0 = the movie itself
        idx = movie[0]
        title = movies[movies.index == idx]["title"].values[0]
        recommendations.append(title)

    return close_match, recommendations


# ---------- UI ----------
st.title("🎬 Movie Recommendation System")
st.write("Type a movie you like, and get similar movie suggestions.")

movies, combined_features = load_data()
similarity = build_similarity_matrix(combined_features)

movie_name = st.text_input("Enter a movie name:", placeholder="e.g. Inception")

top_n = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)

if st.button("Get Recommendations", type="primary"):
    if not movie_name.strip():
        st.warning("Please enter a movie name first.")
    else:
        close_match, recommendations = get_recommendations(movie_name, movies, similarity, top_n=top_n)

        if close_match is None:
            st.error("Couldn't find a close match for that movie. Try checking the spelling.")
        else:
            st.success(f"Showing results for: **{close_match}**")
            for i, title in enumerate(recommendations, start=1):
                st.write(f"{i}. {title}")

st.divider()
st.caption("Built with Streamlit • Data: TMDB Movies Dataset (Kaggle)")
