import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Executive ML & DL Portfolio Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Dark Glassmorphism Theme) ----------
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

    /* Calm Dark Sidebar Social Link Buttons */
    .sidebar-social-btn {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #f8fafc !important;
        font-size: 0.82rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: all 0.2s ease;
        margin-bottom: 0.5rem;
    }
    .sidebar-social-btn:hover {
        border-color: #334155;
        background: rgba(30, 41, 59, 0.8);
        color: #818cf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Canonical Project Directory Data ----------
ML_PROJECTS = [
    {
        "name": "1. Movie Recommendation System",
        "desc": "Content-based movie recommender using TF-IDF feature extraction and cosine similarity on TMDB data.",
        "status": "Live",
        "page": "pages/01_Movie_Recommendation_System.py",
        "icon": "🎬",
        "domain": "Content Recommender",
        "color": "#a855f7",
        "gradient": "linear-gradient(90deg, #7c3aed 0%, #a855f7 100%)",
    },
    {
        "name": "2. Housing Price Prediction",
        "desc": "Regression model estimating Delhi property prices based on area, location, and amenities.",
        "status": "Live",
        "page": "pages/02_Housing_Price_Prediction.py",
        "icon": "🏠",
        "domain": "Real Estate Valuation",
        "color": "#10b981",
        "gradient": "linear-gradient(90deg, #059669 0%, #10b981 100%)",
    },
    {
        "name": "3. Spotify Song Recommendation System",
        "desc": "Music recommendation system matching track audio features using scaling and cosine similarity.",
        "status": "Live",
        "page": "pages/03_Spotify_Song_Recommendation_System.py",
        "icon": "🎵",
        "domain": "Audio Clustering",
        "color": "#34d399",
        "gradient": "linear-gradient(90deg, #059669 0%, #34d399 100%)",
    },
    {
        "name": "4. Email Spam Detector",
        "desc": "Text classification pipeline detecting spam vs. legitimate messages using NLP techniques.",
        "status": "Live",
        "page": "pages/04_Email_Spam_Detector.py",
        "icon": "📧",
        "domain": "NLP / Classification",
        "color": "#3b82f6",
        "gradient": "linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%)",
    },
    {
        "name": "5. Customer Churn Prediction",
        "desc": "Classification model identifying customers likely to stop using a service.",
        "status": "Live",
        "page": "pages/05_Customer_Churn_Prediction.py",
        "icon": "👥",
        "domain": "Customer Analytics",
        "color": "#f59e0b",
        "gradient": "linear-gradient(90deg, #d97706 0%, #f59e0b 100%)",
    },
    {
        "name": "6. Cardiovascular Disease Prediction",
        "desc": "Predictive model assessing cardiovascular disease risk from health indicators.",
        "status": "Live",
        "page": "pages/06_Cardiovascular_Disease_Prediction.py",
        "icon": "❤️",
        "domain": "Clinical Diagnostics",
        "color": "#f43f5e",
        "gradient": "linear-gradient(90deg, #be123c 0%, #f43f5e 100%)",
    },
    {
        "name": "7. Heart Disease Prediction",
        "desc": "Classification model predicting heart disease presence from clinical data.",
        "status": "Live",
        "page": "pages/07_Heart_Disease_Prediction.py",
        "icon": "🫀",
        "domain": "Healthcare Analytics",
        "color": "#e11d48",
        "gradient": "linear-gradient(90deg, #9f1239 0%, #e11d48 100%)",
    },
    {
        "name": "8. Salary Prediction",
        "desc": "Regression model estimating salary based on experience and job-related features.",
        "status": "Live",
        "page": "pages/08_Salary_Prediction.py",
        "icon": "💰",
        "domain": "Workforce Econometrics",
        "color": "#eab308",
        "gradient": "linear-gradient(90deg, #ca8a04 0%, #eab308 100%)",
    },
]

DL_PROJECTS = [
    {
        "name": "9. Pet Breed Classifier",
        "desc": "Deep learning model classifying 46 cat and dog breeds using EfficientNet-B0 transfer learning.",
        "status": "Live",
        "page": "pages/09_Pet_Breed_Classifier.py",
        "icon": "🐾",
        "domain": "Deep Learning / CNN",
        "color": "#2dd4bf",
        "gradient": "linear-gradient(90deg, #0f766e 0%, #14b8a6 100%)",
    },
    {
        "name": "10. Face Detection",
        "desc": "Computer vision model detecting faces in images using OpenCV ResNet-10 SSD.",
        "status": "Live",
        "page": "pages/10_Face_Detection.py",
        "icon": "🙂",
        "domain": "Computer Vision / DNN",
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

GITHUB_URL = "https://github.com/OjasPal/ML-DL-Portfolio"
LINKEDIN_URL = "https://www.linkedin.com/in/ojas-pal/"

# ---------- Sidebar Component ----------
with st.sidebar:
    # High-Fi Portfolio Overview Widget (Formatted flush left to avoid Markdown code block parsing)
    st.markdown(
        """
<div style="background: linear-gradient(145deg, #111827 0%, #0d1322 100%); border: 1px solid #1e293b; border-radius: 10px; padding: 1.15rem 1rem; margin-bottom: 1.5rem; margin-top: 0.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <div style="font-size: 0.72rem; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.5rem;">
        Portfolio Overview
    </div>
    <div style="display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.85rem;">
        <span style="font-size: 1.1rem; margin-top: -0.15rem;">📊</span>
        <div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #f1f5f9;">8 Classical ML Apps</div>
            <div style="font-size: 0.72rem; color: #64748b; line-height: 1.35; margin-top: 0.1rem;">Regression, NLP & Recommenders</div>
        </div>
    </div>
    <div style="display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.85rem;">
        <span style="font-size: 1.1rem; margin-top: -0.15rem;">👁️</span>
        <div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #f1f5f9;">3 Computer Vision Apps</div>
            <div style="font-size: 0.72rem; color: #64748b; line-height: 1.35; margin-top: 0.1rem;">CNNs, ResNet SSD & MLP</div>
        </div>
    </div>
    <div style="display: flex; align-items: flex-start; gap: 0.6rem;">
        <span style="font-size: 1.1rem; margin-top: -0.15rem;">⚡</span>
        <div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #34d399;">100% Pre-compiled</div>
            <div style="font-size: 0.72rem; color: #64748b; line-height: 1.35; margin-top: 0.1rem;">Sub-second cached inference</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Author & Social Buttons
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 0.3rem;">
            <strong>Developer:</strong> Ojas Pal
        </div>
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5; margin-bottom: 1rem;">
            Migrated from individual exploratory Jupyter notebooks into high-performance web applications.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <a href="{GITHUB_URL}" target="_blank" class="sidebar-social-btn">
            <span style="display: flex; align-items: center; gap: 0.5rem;">
                <svg height="15" width="15" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                GitHub Repository
            </span>
            <span>&rarr;</span>
        </a>

        <a href="{LINKEDIN_URL}" target="_blank" class="sidebar-social-btn">
            <span style="display: flex; align-items: center; gap: 0.5rem;">
                <svg height="15" width="15" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/>
                </svg>
                LinkedIn Profile
            </span>
            <span>&rarr;</span>
        </a>
        """,
        unsafe_allow_html=True,
    )

# =========================================================================
# HERO BANNER SECTION
# =========================================================================
st.markdown(
    f"""
    <div class="hero-card">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <span style="font-size: 2.25rem;">🤖</span>
                <h1 style="font-size: 2.35rem; font-weight: 800; color: #f8fafc; margin: 0; letter-spacing: -0.03em;">
                    ML & DL Project Portfolio
                </h1>
            </div>
            <a href="{GITHUB_URL}" target="_blank" style="text-decoration: none;">
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
            <span class="tech-pill">PyTorch / CUDA</span>
            <span class="tech-pill">Scikit-Learn</span>
            <span class="tech-pill">OpenCV</span>
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
    st.metric(label="DEPLOYED LIVE APPS", value="11 Active")
with col_m3:
    st.metric(label="INFERENCE LATENCY", value="< 100 ms")
with col_m4:
    st.metric(label="ML LOGIC PRESERVATION", value="100%")

st.markdown("<div style='height: 2.25rem;'></div>", unsafe_allow_html=True)

# Helper function to render project cards
def render_project_grid(projects_list):
    for i in range(0, len(projects_list), 2):
        batch = projects_list[i : i + 2]
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

# =========================================================================
# SECTION 1: MACHINE LEARNING & ANALYTICS
# =========================================================================
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <h2 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0;">
                📊 Machine Learning & Predictive Analytics
            </h2>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.2rem; margin-bottom: 0;">
                Regression, classification, and recommender systems built with Scikit-Learn and NLTK.
            </p>
        </div>
        <span class="badge-live">8 APPS READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

render_project_grid(ML_PROJECTS)

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# =========================================================================
# SECTION 2: DEEP LEARNING & COMPUTER VISION
# =========================================================================
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <h2 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0;">
                👁️ Deep Learning & Computer Vision
            </h2>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.2rem; margin-bottom: 0;">
                Convolutional neural networks, OpenCV face detectors, and PyTorch transfer learning models.
            </p>
        </div>
        <span class="badge-live">3 APPS READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

render_project_grid(DL_PROJECTS)

# =========================================================================
# FOOTER SECTION
# =========================================================================
st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="border-top: 1px solid #1e293b; padding-top: 1.25rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div style="font-size: 0.82rem; color: #64748b;">
            <span>Built by <strong>Ojas Pal</strong> • Powered by Streamlit, PyTorch & Scikit-Learn</span>
        </div>
        <div style="font-size: 0.82rem; color: #64748b;">
            <span>Single Multi-Page Architecture</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)