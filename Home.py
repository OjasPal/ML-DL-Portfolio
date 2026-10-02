import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Executive ML & DL Portfolio Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Vercel / AI Studio Dark Glassmorphism Theme) ----------
st.markdown(
    """
    <style>
    /* Global Deep Space Obsidian Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Metric Values & Labels */
    [data-testid="stMetricValue"] {
        color: #818cf8 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        text-shadow: 0 0 16px rgba(129, 140, 248, 0.35);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hero Banner Container */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 2rem 2.25rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
    }
    .hero-card::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0) 70%);
        pointer-events: none;
    }

    /* Pill Tags */
    .tech-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        color: #cbd5e1;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Project Cards */
    .project-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.35rem 1.4rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .project-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.5), 0 0 16px rgba(99, 102, 241, 0.15);
    }
    .card-accent-strip {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }

    /* Status Badges */
    .badge-live {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        letter-spacing: 0.04em;
    }
    .badge-soon {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        letter-spacing: 0.04em;
    }

    /* Page link buttons styling */
    div[data-testid="stPageLink"] a {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stPageLink"] a:hover {
        border-color: #818cf8 !important;
        background-color: #111827 !important;
        color: #818cf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Canonical Project Directory Data ----------
PROJECTS = [
    {
        "name": "1. Movie Recommendation System",
        "desc": "Content-based movie recommender using TF-IDF feature extraction and cosine similarity on TMDB data.",
        "status": "Live",
        "page": "pages/1_Movie_Recommendation_System.py",
        "icon": "🎬",
        "domain": "Content Recommender",
        "color": "#a855f7",
        "gradient": "linear-gradient(90deg, #7c3aed 0%, #a855f7 100%)",
    },
    {
        "name": "2. Housing Price Prediction",
        "desc": "Regression model estimating Delhi property prices based on area, location, and amenities.",
        "status": "Live",
        "page": "pages/2_Housing_Price_Prediction.py",
        "icon": "🏠",
        "domain": "Real Estate Valuation",
        "color": "#10b981",
        "gradient": "linear-gradient(90deg, #059669 0%, #10b981 100%)",
    },
    {
        "name": "3. Spotify Song Recommendation System",
        "desc": "Music recommendation system matching track audio features using scaling and cosine similarity.",
        "status": "Live",
        "page": "pages/3_Spotify_Song_Recommendation_System.py",
        "icon": "🎵",
        "domain": "Audio Clustering",
        "color": "#34d399",
        "gradient": "linear-gradient(90deg, #059669 0%, #34d399 100%)",
    },
    {
        "name": "4. Email Spam Detector",
        "desc": "Text classification pipeline detecting spam vs. legitimate messages using NLP techniques.",
        "status": "Live",
        "page": "pages/4_Email_Spam_Detector.py",
        "icon": "📧",
        "domain": "NLP / Classification",
        "color": "#3b82f6",
        "gradient": "linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%)",
    },
    {
        "name": "5. Customer Churn Prediction",
        "desc": "Classification model identifying customers likely to stop using a service.",
        "status": "Live",
        "page": "pages/5_Customer_Churn_Prediction.py",
        "icon": "👥",
        "domain": "Customer Analytics",
        "color": "#f59e0b",
        "gradient": "linear-gradient(90deg, #d97706 0%, #f59e0b 100%)",
    },
    {
        "name": "6. Cardiovascular Disease Prediction",
        "desc": "Predictive model assessing cardiovascular disease risk from health indicators.",
        "status": "Live",
        "page": "pages/6_Cardiovascular_Disease_Prediction.py",
        "icon": "❤️",
        "domain": "Clinical Diagnostics",
        "color": "#f43f5e",
        "gradient": "linear-gradient(90deg, #be123c 0%, #f43f5e 100%)",
    },
    {
        "name": "7. Heart Disease Prediction",
        "desc": "Classification model predicting heart disease presence from clinical data.",
        "status": "Live",
        "page": "pages/7_Heart_Disease_Prediction.py",
        "icon": "🫀",
        "domain": "Healthcare Analytics",
        "color": "#e11d48",
        "gradient": "linear-gradient(90deg, #9f1239 0%, #e11d48 100%)",
    },
    {
        "name": "8. Salary Prediction",
        "desc": "Regression model estimating salary based on experience and job-related features.",
        "status": "Coming Soon",
        "page": None,
        "icon": "💰",
        "domain": "Workforce Econometrics",
        "color": "#eab308",
        "gradient": "linear-gradient(90deg, #ca8a04 0%, #eab308 100%)",
    },
    {
        "name": "9. Image Classification",
        "desc": "Deep learning model classifying images into multiple categories.",
        "status": "Coming Soon",
        "page": None,
        "icon": "🖼️",
        "domain": "Deep Learning / CNN",
        "color": "#2dd4bf",
        "gradient": "linear-gradient(90deg, #0f766e 0%, #14b8a6 100%)",
    },
    {
        "name": "10. Face Detection",
        "desc": "Computer vision model detecting faces within images.",
        "status": "Coming Soon",
        "page": None,
        "icon": "🙂",
        "domain": "Computer Vision",
        "color": "#818cf8",
        "gradient": "linear-gradient(90deg, #4338ca 0%, #6366f1 100%)",
    },
    {
        "name": "11. Handwritten Digit Recognizer",
        "desc": "Neural network classifying hand-drawn digits in real time, trained on the MNIST dataset.",
        "status": "Live",
        "page": "pages/11_Handwritten_Digit_Recognizer.py",
        "icon": "✍️",
        "domain": "Computer Vision / MLP",
        "color": "#38bdf8",
        "gradient": "linear-gradient(90deg, #0284c7 0%, #38bdf8 100%)",
    },
]

# ---------- Sidebar Component ----------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.25rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span style="font-size: 1.5rem;">🤖</span>
                <span style="font-size: 1.1rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                    ML-DL Portfolio
                </span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.4;">
                Interactive Multi-Page Hub
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                PORTFOLIO HEALTH
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-size: 0.82rem; color: #cbd5e1;">Live Interactive Apps</span>
                <span style="font-size: 0.85rem; font-weight: 700; color: #34d399; font-family: monospace;">8 / 11</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.82rem; color: #cbd5e1;">In Active Pipeline</span>
                <span style="font-size: 0.85rem; font-weight: 700; color: #fbbf24; font-family: monospace;">3 / 11</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.6; margin-bottom: 1.5rem;">
            👈 <strong>Navigation:</strong> Use Streamlit's page tree above to test individual models with live inputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Author & GitHub Badge
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 0.5rem;">
            <strong>Developer:</strong> Ojas Pal
        </div>
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5; margin-bottom: 1rem;">
            Migrated from individual exploratory Jupyter notebooks into high-performance web applications.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <a href="https://github.com/OjasPal/ML-DL-Portfolio" target="_blank" style="text-decoration: none;">
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 0.65rem 0.85rem; display: flex; align-items: center; justify-content: space-between; color: #f8fafc; font-size: 0.82rem; font-weight: 600; transition: border-color 0.2s;">
                <span style="display: flex; align-items: center; gap: 0.5rem;">
                    <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    GitHub Repository
                </span>
                <span style="color: #818cf8;">&rarr;</span>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

# =========================================================================
# HERO BANNER SECTION
# =========================================================================
st.markdown(
    """
    <div class="hero-card">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <span style="font-size: 2.25rem;">🤖</span>
                <h1 style="font-size: 2.35rem; font-weight: 800; color: #f8fafc; margin: 0; letter-spacing: -0.03em;">
                    ML & DL Project Portfolio
                </h1>
            </div>
            <a href="https://github.com/OjasPal/ML-DL-Portfolio" target="_blank" style="text-decoration: none;">
                <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid #6366f1; color: #c7d2fe; font-size: 0.82rem; font-weight: 600; padding: 0.45rem 1rem; border-radius: 8px; display: inline-flex; align-items: center; gap: 0.5rem; transition: background 0.2s;">
                    <svg height="15" width="15" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    <span>View Repository on GitHub</span>
                    <span style="font-size: 0.9rem;">&nearr;</span>
                </div>
            </a>
        </div>
        <p style="color: #94a3b8; font-size: 1.02rem; max-width: 820px; line-height: 1.6; margin-top: 0.5rem; margin-bottom: 1.25rem;">
            An executive showcase of <strong>11 machine learning and deep learning solutions</strong>. Originally developed as 
            exploratory Jupyter notebooks, each project is refactored into a high-performance, single-application multi-page 
            Streamlit dashboard featuring pre-compiled artifact caching (<code>@st.cache_resource</code>) for sub-second live inference.
        </p>
        <div>
            <span class="tech-pill">Python 3.12</span>
            <span class="tech-pill">Streamlit Multi-Page</span>
            <span class="tech-pill">Scikit-Learn</span>
            <span class="tech-pill">TensorFlow / Keras</span>
            <span class="tech-pill">Joblib Pre-caching</span>
            <span class="tech-pill">NLTK NLP</span>
            <span class="tech-pill">Pandas & NumPy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Metrics Bar ----------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="TOTAL ARCHITECTURES", value="11 Projects")
with col_m2:
    st.metric(label="DEPLOYED LIVE APPS", value="8 Active")
with col_m3:
    st.metric(label="INFERENCE LATENCY", value="< 100 ms")
with col_m4:
    st.metric(label="ML LOGIC PRESERVATION", value="100%")

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# =========================================================================
# FEATURED LIVE PROJECTS SHOWCASE
# =========================================================================
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <h2 style="font-size: 1.45rem; font-weight: 700; color: #f8fafc; margin: 0;">
                ⚡ Featured Live Models
            </h2>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.2rem; margin-bottom: 0;">
                Fully deployed, interactive models with real-time prediction and pre-compiled model artifacts.
            </p>
        </div>
        <span class="badge-live">
            <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; display: inline-block;"></span>
            8 APPS READY
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

live_projects = [p for p in PROJECTS if p["status"] == "Live"]

# Render live projects in responsive column pairs
for i in range(0, len(live_projects), 2):
    batch = live_projects[i : i + 2]
    cols = st.columns(len(batch), gap="medium")
    for j, project in enumerate(batch):
        with cols[j]:
            st.markdown(
                f"""
                <div class="project-card">
                    <div class="card-accent-strip" style="background: {project['gradient']};"></div>
                    <div>
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                            <span style="font-size: 1.75rem;">{project['icon']}</span>
                            <span class="badge-live">● LIVE</span>
                        </div>
                        <div style="font-size: 0.75rem; font-weight: 600; color: {project['color']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
                            {project['domain']}
                        </div>
                        <h3 style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-top: 0; margin-bottom: 0.55rem; line-height: 1.35;">
                            {project['name']}
                        </h3>
                        <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.55; margin-bottom: 1.25rem;">
                            {project['desc']}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Direct Streamlit Page Link
            if project["page"]:
                try:
                    st.page_link(
                        project["page"],
                        label=f"Launch {project['name'].split('. ')[1]}",
                        icon=project["icon"],
                    )
                except Exception:
                    st.caption("Select this model from the left sidebar.")
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# =========================================================================
# UPCOMING RELEASES (IN DEVELOPMENT)
# =========================================================================
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <h2 style="font-size: 1.45rem; font-weight: 700; color: #f8fafc; margin: 0;">
                🔜 Upcoming Releases (In Development)
            </h2>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.2rem; margin-bottom: 0;">
                Original Jupyter notebooks currently queued for Streamlit UI adaptation and artifact compilation.
            </p>
        </div>
        <span class="badge-soon">
            3 PIPELINE QUEUED
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

upcoming_projects = [p for p in PROJECTS if p["status"] == "Coming Soon"]

for i in range(0, len(upcoming_projects), 3):
    batch = upcoming_projects[i : i + 3]
    cols = st.columns(3, gap="medium")
    for j, project in enumerate(batch):
        with cols[j]:
            st.markdown(
                f"""
                <div class="project-card" style="opacity: 0.85; border-color: #1e293b;">
                    <div class="card-accent-strip" style="background: #334155;"></div>
                    <div>
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                            <span style="font-size: 1.5rem;">{project['icon']}</span>
                            <span class="badge-soon">COMING SOON</span>
                        </div>
                        <div style="font-size: 0.72rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;">
                            {project['domain']}
                        </div>
                        <h4 style="font-size: 1.02rem; font-weight: 700; color: #f1f5f9; margin-top: 0; margin-bottom: 0.45rem;">
                            {project['name']}
                        </h4>
                        <p style="color: #64748b; font-size: 0.82rem; line-height: 1.5; margin-bottom: 0.75rem;">
                            {project['desc']}
                        </p>
                    </div>
                    <div style="font-size: 0.75rem; color: #475569; font-style: italic; border-top: 1px solid #1e293b; padding-top: 0.65rem;">
                        ⚡ Migration Next in Queue
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# =========================================================================
# FOOTER SECTION
# =========================================================================
st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="border-top: 1px solid #1e293b; padding-top: 1.25rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div style="font-size: 0.82rem; color: #64748b;">
            <span>Built by <strong>Ojas Pal</strong> • Powered by Streamlit & Scikit-Learn</span>
        </div>
        <div style="font-size: 0.82rem; color: #64748b;">
            <span>Single Multi-Page Architecture</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)