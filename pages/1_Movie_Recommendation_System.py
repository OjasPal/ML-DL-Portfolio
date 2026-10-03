import difflib
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page Config ----------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Neon Purple Accent) ----------
st.markdown(
    """
    <style>
    /* Global Deep Space Theme */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Streamlit Card & Expander Container */
    div[data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Metric Values & Labels - Neon Purple Glow */
    [data-testid="stMetricValue"] {
        color: #a855f7 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(168, 85, 247, 0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Input & Select Elements */
    .stTextInput input, .stTextArea textarea {
        background-color: #0d1322 !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 1px #a855f7 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Neon Purple Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #7c3aed 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #6d28d9 0%, #9333ea 100%) !important;
        box-shadow: 0 6px 22px rgba(168, 85, 247, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Info Sidebar Cards */
    .ais-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .ais-card h4 {
        color: #f8fafc;
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Recommendation Cards */
    .rec-item {
        background-color: #0d1322;
        border: 1px solid #1e293b;
        border-left: 3px solid #a855f7;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: border-color 0.2s ease;
    }
    .rec-item:hover {
        border-color: #a855f7;
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Paths Definition ----------
PROJECT_DIR = "projects/movie-recommendation-system"
MOVIES_PKL_PATH = os.path.join(PROJECT_DIR, "models", "movies_data.pkl")
VECTORS_PKL_PATH = os.path.join(PROJECT_DIR, "models", "feature_vectors.pkl")

FALLBACK_MOVIES_PATHS = [
    MOVIES_PKL_PATH,
    "models/movies_data.pkl",
    "movies_data.pkl",
]
FALLBACK_VECTORS_PATHS = [
    VECTORS_PKL_PATH,
    "models/feature_vectors.pkl",
    "feature_vectors.pkl",
]


# ---------- Lightweight Model Artifact Loading ----------
@st.cache_resource(show_spinner="Loading lightweight movie feature artifacts...")
def load_artifacts():
    movies_path = next((p for p in FALLBACK_MOVIES_PATHS if os.path.exists(p)), None)
    vectors_path = next((p for p in FALLBACK_VECTORS_PATHS if os.path.exists(p)), None)

    if movies_path and vectors_path:
        movies = joblib.load(movies_path)
        feature_vectors = joblib.load(vectors_path)
        return movies, feature_vectors
    return None, None


def get_recommendations(movie_name, movies, feature_vectors, top_n=30):
    list_of_all_titles = movies["title"].dropna().tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return None, []

    close_match = find_close_match[0]
    index_of_the_movie = movies[movies.title == close_match]["index"].values[0]

    # Calculate Cosine Similarity on-the-fly for the selected query vector (Instant & Zero RAM bloat)
    query_vector = feature_vectors[index_of_the_movie]
    similarity_score = cosine_similarity(query_vector, feature_vectors).flatten()

    similarity_score_list = list(enumerate(similarity_score))
    sorted_similar_movies = sorted(similarity_score_list, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie in sorted_similar_movies[1 : top_n + 1]:  # skip index 0 = the movie itself
        idx = movie[0]
        title = movies[movies.index == idx]["title"].values[0]
        recommendations.append(title)

    return close_match, recommendations


# ---------- Resolve & Load Artifacts ----------
movies, feature_vectors = load_artifacts()

# ---------- Layout Grid: Main Column (2.2) & Info Sidebar (1.0) ----------
col_main, col_info = st.columns([2.2, 1.0], gap="large")

# =========================================================================
# MAIN COLUMN
# =========================================================================
with col_main:
    # Header Section
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 2rem;">🎬</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Movie Recommendation System
                </h1>
                <span style="background: rgba(168, 85, 247, 0.15); border: 1px solid #a855f7; color: #d8b4fe; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 1
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Discover tailored movie recommendations computed using a Content-Based Filtering algorithm 
                with <strong>TF-IDF Vectorization</strong> and <strong>Cosine Similarity</strong> across genres, keywords, overviews, and taglines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if movies is None or feature_vectors is None:
        st.error(
            f"Pre-compiled feature binaries not found at `{MOVIES_PKL_PATH}`! "
            "Please run your Jupyter notebook to export `movies_data.pkl` and `feature_vectors.pkl` first."
        )
        st.stop()

    # Model & Dataset Summary Card
    with st.expander("⚡ Model Architecture & Dataset Metrics", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Content-Based Cosine Similarity computed on-the-fly across multi-feature TF-IDF representations."
            "</p>",
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="TOTAL MOVIES INDEXED", value=f"{len(movies):,}")
        with col_m2:
            st.metric(label="SIMILARITY ALGORITHM", value="Cosine Similarity")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Movie Query Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Interactive Movie Recommendation"
        "</h3>",
        unsafe_allow_html=True,
    )

    sample_options = [
        "Custom Movie Title",
        "Inception",
        "Avatar",
        "The Dark Knight",
        "Interstellar",
        "The Avengers",
        "Pulp Fiction",
    ]

    selected_sample = st.selectbox(
        "Choose a popular movie preset or type below:",
        options=sample_options,
        index=1,
    )

    default_title = "" if selected_sample == "Custom Movie Title" else selected_sample

    movie_name = st.text_input(
        "Enter a movie name:",
        value=default_title,
        placeholder="e.g. Inception, The Matrix, Gladiator...",
    )

    top_n = st.slider(
        "Number of recommendations to fetch",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )

    get_rec_clicked = st.button("⚡ Get Recommendations")

    if get_rec_clicked:
        if not movie_name.strip():
            st.warning("Please enter a movie name first.")
        else:
            close_match, recommendations = get_recommendations(
                movie_name, movies, feature_vectors, top_n=top_n
            )

            if close_match is None:
                st.error("Couldn't find a close match for that movie. Try checking the spelling.")
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(124, 58, 237, 0.18); border: 1px solid #a855f7; border-radius: 10px; padding: 1.15rem; margin-top: 1.25rem; margin-bottom: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 0.65rem;">
                                <span style="font-size: 1.5rem;">🎯</span>
                                <span style="font-size: 1.15rem; font-weight: 700; color: #f3e8ff;">Matched Title: {close_match}</span>
                            </div>
                            <span style="background: linear-gradient(90deg, #7c3aed 0%, #a855f7 100%); color: #ffffff; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.82rem; font-family: monospace;">
                                Top {len(recommendations)} Recommendations
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<h4 style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.75rem;'>"
                    "Similar Movies You Might Like:"
                    "</h4>",
                    unsafe_allow_html=True,
                )

                for i, title in enumerate(recommendations, start=1):
                    st.markdown(
                        f"""
                        <div class="rec-item">
                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                <span style="background: rgba(168, 85, 247, 0.2); color: #d8b4fe; font-weight: 700; font-size: 0.8rem; border-radius: 6px; padding: 0.15rem 0.5rem; font-family: monospace;">
                                    #{i:02d}
                                </span>
                                <span style="font-size: 0.92rem; font-weight: 500; color: #f8fafc;">
                                    {title}
                                </span>
                            </div>
                            <span style="color: #a855f7; font-size: 0.8rem;">🎬</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • TF-IDF Vectorizer & Cosine Similarity</span>"
        "<span>Data: TMDB Movies Dataset (Kaggle)</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# =========================================================================
# RIGHT-HAND INFO COLUMN (SIDEBAR / DOCUMENTATION CARD)
# =========================================================================
with col_info:
    # Notebook to Streamlit Conversion Card
    st.markdown(
        """
        <div class="ais-card">
            <h4>
                <span>⚙️</span> Notebook to Streamlit Conversion
            </h4>
            <ul style="color: #94a3b8; font-size: 0.82rem; line-height: 1.6; padding-left: 1.15rem; margin-bottom: 0;">
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_resource:</strong> Loads lightweight <code>movies_data.pkl</code> and <code>feature_vectors.pkl</code> binaries via <strong>joblib</strong> for zero-latency startup.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">On-The-Fly Similarity:</strong> Computes Cosine Similarity dynamically per query vector, avoiding the 18GB+ matrix memory footprint.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">difflib.get_close_matches:</strong> Employs fuzzy string matching to resolve misspellings to database titles.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Relative Artifact Path:</strong> Loads binary models directly from <code>projects/movie-recommendation-system/models/</code>.
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Kaggle Downloader Info Card
    st.markdown(
        """
        <div class="ais-card">
            <h4>
                <span>📥</span> Kaggle Downloader Info
            </h4>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                <div style="margin-bottom: 0.6rem;">
                    <span style="color: #cbd5e1; font-weight: 500;">Dataset Target:</span><br/>
                    <code style="color: #a855f7; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">tmdb_movies_50k.csv</code>
                </div>
                <div style="margin-bottom: 0.6rem;">
                    <span style="color: #cbd5e1; font-weight: 500;">CLI Command:</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        "kaggle datasets download -d asaniczka/tmdb-movies-dataset-2023-930k-movies -p projects/movie-recommendation-system/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 <code>download_data.py</code> detects existing data in the target folder and skips redundant downloads.
        </p>
        """,
        unsafe_allow_html=True,
    )