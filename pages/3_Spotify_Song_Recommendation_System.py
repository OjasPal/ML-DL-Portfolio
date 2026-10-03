import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Spotify Song Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Spotify Mint Accent) ----------
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

    /* Metric Values & Labels - Spotify Mint Glow */
    [data-testid="stMetricValue"] {
        color: #34d399 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(52, 211, 153, 0.4);
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
        border-color: #34d399 !important;
        box-shadow: 0 0 0 1px #34d399 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Spotify Mint Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #059669 0%, #34d399 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #047857 0%, #10b981 100%) !important;
        box-shadow: 0 6px 22px rgba(52, 211, 153, 0.5) !important;
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

    /* Track Cards */
    .track-item {
        background-color: #0d1322;
        border: 1px solid #1e293b;
        border-left: 3px solid #34d399;
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: border-color 0.2s ease;
    }
    .track-item:hover {
        border-color: #34d399;
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Paths Definition & Search Hierarchy ----------
PROJECT_DIR = "projects/spotify-song-recommendation-system"
CANDIDATE_DIRS = [
    os.path.join(PROJECT_DIR, "model"),
    os.path.join(PROJECT_DIR, "models"),
    "model",
    "models",
    PROJECT_DIR,
    ".",
]


def resolve_file(filename):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


DF_PATH = resolve_file("spotify_df.joblib")
SIMILARITY_PATH = resolve_file("spotify_similarity.joblib")
SCALER_PATH = resolve_file("spotify_scaler.joblib")
ENCODERS_PATH = resolve_file("spotify_encoders.joblib")

# Fallback sample library if joblib models are pending download
SAMPLE_FALLBACK_TRACKS = pd.DataFrame(
    [
        {"track_name": "Blinding Lights", "artist_name": "The Weeknd", "album_name": "After Hours", "genre": "Synthpop"},
        {"track_name": "Save Your Tears", "artist_name": "The Weeknd", "album_name": "After Hours", "genre": "Synthpop"},
        {"track_name": "Levitating", "artist_name": "Dua Lipa", "album_name": "Future Nostalgia", "genre": "Dance-Pop"},
        {"track_name": "Don't Start Now", "artist_name": "Dua Lipa", "album_name": "Future Nostalgia", "genre": "Nu-Disco"},
        {"track_name": "Shape of You", "artist_name": "Ed Sheeran", "album_name": "÷ (Divide)", "genre": "Pop"},
        {"track_name": "Bad Habits", "artist_name": "Ed Sheeran", "album_name": "= (Equals)", "genre": "Dance-Pop"},
        {"track_name": "Starboy", "artist_name": "The Weeknd, Daft Punk", "album_name": "Starboy", "genre": "Electro-Pop"},
        {"track_name": "As It Was", "artist_name": "Harry Styles", "album_name": "Harry's House", "genre": "Indie Pop"},
        {"track_name": "Watermelon Sugar", "artist_name": "Harry Styles", "album_name": "Fine Line", "genre": "Pop Rock"},
        {"track_name": "Stay", "artist_name": "The Kid LAROI, Justin Bieber", "album_name": "F*CK LOVE 3", "genre": "Pop"},
        {"track_name": "Peaches", "artist_name": "Justin Bieber, Daniel Caesar", "album_name": "Justice", "genre": "R&B"},
        {"track_name": "Someone You Loved", "artist_name": "Lewis Capaldi", "album_name": "Divinely Uninspired", "genre": "Pop"},
    ]
)


# ---------------------------------------------------------
# Load Predefined Models & Artifacts (Instant Cache)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading track dataset...")
def load_data():
    if DF_PATH:
        return joblib.load(DF_PATH), True
    return SAMPLE_FALLBACK_TRACKS, False


@st.cache_resource(show_spinner="Loading audio similarity models & encoders...")
def load_artifacts():
    if SIMILARITY_PATH and SCALER_PATH and ENCODERS_PATH:
        similarity = joblib.load(SIMILARITY_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return similarity, scaler, encoders, True
    return None, None, None, False


df, df_loaded_from_joblib = load_data()
similarity, scaler, encoders, models_loaded = load_artifacts()


# ---------------------------------------------------------
# Helper Function: Extract Artist Name
# ---------------------------------------------------------
def extract_artist(row):
    artist = row.get("track_artist")
    if pd.notna(artist) and str(artist).strip() != "":
        return str(artist).strip()
    return "Unknown Artist"


# ---------------------------------------------------------
# Recommendation Logic (Preserving Original ML Architecture)
# ---------------------------------------------------------
def get_recommendations(song_title, top_n=5):
    # Case-insensitive title match
    matches = df[df["track_name"].astype(str).str.lower() == song_title.lower().strip()]

    # If exact match isn't found, try partial match
    if len(matches) == 0:
        matches = df[df["track_name"].astype(str).str.lower().str.contains(song_title.lower().strip(), regex=False)]

    if len(matches) == 0:
        return None

    idx = matches.index[0]

    # If full similarity matrix is loaded, compute top similarity neighbors
    if similarity is not None:
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the selected song itself
        sim_scores = sim_scores[1 : top_n + 1]
        song_indices = [i[0] for i in sim_scores]
        return df.iloc[song_indices]
    else:
        # Fallback simulation if running before dataset download
        selected_genre = df.loc[idx].get("genre", "")
        same_genre = df[df.index != idx]
        if "genre" in df.columns and selected_genre:
            matches_genre = same_genre[same_genre["genre"] == selected_genre]
            if len(matches_genre) < top_n:
                matches_genre = pd.concat([matches_genre, same_genre[same_genre["genre"] != selected_genre]])
            return matches_genre.head(top_n)
        return same_genre.head(top_n)


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
                <span style="font-size: 2rem;">🎵</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Spotify Song Recommendation System
                </h1>
                <span style="background: rgba(52, 211, 153, 0.15); border: 1px solid #10b981; color: #6ee7b7; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 3
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Select or search for any song to discover acoustic lookalikes powered by 
                <strong>StandardScaler</strong> normalization and high-dimensional <strong>Cosine Similarity</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Performance & Library Metrics Card
    with st.expander("⚡ Model Architecture & Audio Feature Metrics", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Acoustic vector representation evaluated on danceability, energy, tempo, loudness, and valence."
            "</p>",
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="TRACKS IN ARCHIVE", value=f"{len(df):,}")
        with col_m2:
            st.metric(label="SIMILARITY ENGINE", value="Cosine Distance Matrix")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Track Selection Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Search & Generate Recommendations"
        "</h3>",
        unsafe_allow_html=True,
    )

    # Search Bar for Custom Song
    custom_song_input = st.text_input(
        "🔎 Enter your favorite song title:",
        placeholder="e.g., Blinding Lights, Shape of You, Starboy...",
    )

    # Limit Dropdown Options to 10
    top_10_tracks = df["track_name"].dropna().unique()[:10].tolist()
    selected_from_dropdown = st.selectbox(
        "Or pick from top 10 popular tracks:",
        options=["-- Select a sample track --"] + top_10_tracks,
        index=0,
    )

    # Determine which track input to prioritize
    song_to_query = ""
    if custom_song_input.strip():
        song_to_query = custom_song_input.strip()
    elif selected_from_dropdown != "-- Select a sample track --":
        song_to_query = selected_from_dropdown

    num_recommendations = st.slider(
        "Number of recommendations to fetch:",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    get_recs_clicked = st.button("⚡ Get Recommendations")

    if get_recs_clicked:
        if not song_to_query:
            st.warning("Please enter a song name in the search bar or choose one from the top 10 dropdown.")
        else:
            with st.spinner(f"Finding acoustic neighbors for '{song_to_query}'..."):
                recommendations = get_recommendations(song_to_query, top_n=num_recommendations)

                if recommendations is not None and not recommendations.empty:
                    st.markdown(
                        f"""
                            <div style="background: rgba(5, 150, 105, 0.2); border: 1px solid #34d399; border-radius: 10px; padding: 1.15rem; margin-top: 1.25rem; margin-bottom: 1.25rem;">
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                                    <div style="display: flex; align-items: center; gap: 0.65rem;">
                                        <span style="font-size: 1.5rem;">🎧</span>
                                        <span style="font-size: 1.15rem; font-weight: 700; color: #ecfdf5;">Seed Track: {song_to_query}</span>
                                    </div>
                                    <span style="background: linear-gradient(90deg, #059669 0%, #34d399 100%); color: #ffffff; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.82rem; font-family: monospace;">
                                        Top {len(recommendations)} Acoustic Matches
                                    </span>
                                </div>
                            </div>
                            """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "<h4 style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.75rem;'>"
                        "Recommended Tracks:"
                        "</h4>",
                        unsafe_allow_html=True,
                    )

                    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
                        t_name = row.get("track_name", "Unknown Title")
                        a_name = extract_artist(row)
                        alb_name = row.get("track_album_name", "")
                        g_name = row.get("playlist_genre", "")

                        sub_info = f"by <strong>{a_name}</strong>"
                        if alb_name and pd.notna(alb_name):
                            sub_info += f" • <em>{alb_name}</em>"
                        if g_name and pd.notna(g_name):
                            sub_info += f" • <span style='color: #6ee7b7;'>[{g_name}]</span>"

                        # Ensure this st.markdown block matches the 24-space / 6-tab indent level inside the loop
                        st.markdown(
                            f"""
                                <div class="track-item">
                                    <div style="display: flex; align-items: center; gap: 0.85rem;">
                                        <span style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; font-weight: 700; font-size: 0.82rem; border-radius: 6px; padding: 0.2rem 0.55rem; font-family: monospace;">
                                            #{i:02d}
                                        </span>
                                        <div>
                                            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">
                                                {t_name}
                                            </div>
                                            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.1rem;">
                                                {sub_info}
                                            </div>
                                        </div>
                                    </div>
                                    <span style="color: #34d399; font-size: 0.85rem;">🎵</span>
                                </div>
                                """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning(
                        f"Could not find '{song_to_query}' in the music library. Try typing a different song title or selecting from the dropdown.")

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • Pre-compiled Joblib Similarity & StandardScaler</span>"
        "<span>Data: Spotify Audio Features Dataset (Kaggle)</span>"
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
                    <strong style="color: #e2e8f0;">@st.cache_resource:</strong> Pre-trained artifacts 
                    (<code>spotify_similarity.joblib</code>, <code>spotify_scaler.joblib</code>, and <code>spotify_encoders.joblib</code>) 
                    are loaded via <strong>joblib</strong> for zero-latency inference.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_data:</strong> Caches track catalog (<code>spotify_df.joblib</code>) in memory.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Case-Insensitive Querying:</strong> Track titles are matched using lowered strings, ensuring robustness to casing differences.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Path Hierarchy:</strong> Checks <code>projects/spotify-song-recommendation-system/model/</code> and local directories seamlessly.
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
                    <code style="color: #34d399; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">spotify_df.joblib</code>
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
        "kaggle datasets download -d joebeachcapital/30000-spotify-songs -p projects/spotify-song-recommendation-system/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 Model artifacts in <code>model/</code> load instantaneously, keeping your app lightweight and responsive.
        </p>
        """,
        unsafe_allow_html=True,
    )