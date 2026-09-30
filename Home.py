import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="ML & DL Project Portfolio",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Header & Introduction ----------
st.title("🤖 ML & DL Project Portfolio")
st.write(
    """
    Welcome to my machine learning and deep learning portfolio! This is a collection of 
    learning projects originally developed as exploratory Jupyter notebooks and now converted 
    into an interactive, multi-page web application.
    """
)

st.info("👈 **How to navigate:** Select any active project from the sidebar on the left to test the model and run predictions live.")

st.divider()

# ---------- Projects Directory Data ----------
# Easily update any project's status from "Coming Soon" to "Live" and provide its page path.
# NOTE: descriptions below are placeholders for not-yet-converted projects — update each
# one line once you actually revisit that notebook.
PROJECTS = [
    {
        "name": "1. Movie Recommendation System",
        "desc": "Content-based movie recommender using TF-IDF feature extraction and cosine similarity on TMDB data.",
        "status": "Live",
        "page": "pages/1_Movie_Recommendation_System.py",
        "icon": "🎬",
    },
    {
        "name": "2. Housing Price Prediction",
        "desc": "Regression model estimating Delhi property prices based on area, location, and amenities.",
        "status": "Live",
        "page": "pages/2_Housing_Price_Prediction.py",
        "icon": "🏠",
    },
    {
        "name": "3. Spotify Song Recommendation System",
        "desc": "Music recommendation system matching track audio features using scaling and cosine similarity.",
        "status": "Live",
        "page": "pages/3_Spotify_Song_Recommendation_System.py",
        "icon": "🎵",
    },
    {
        "name": "4. Email Spam Detector",
        "desc": "Text classification pipeline detecting spam vs. legitimate messages using NLP techniques.",
        "status": "Coming Soon",
        "page": None,
        "icon": "📧",
    },
    {
        "name": "5. Customer Churn Prediction",
        "desc": "Classification model identifying customers likely to stop using a service.",
        "status": "Coming Soon",
        "page": None,
        "icon": "👥",
    },
    {
        "name": "6. Cardiovascular Disease Prediction",
        "desc": "Predictive model assessing cardiovascular disease risk from health indicators.",
        "status": "Coming Soon",
        "page": None,
        "icon": "❤️",
    },
    {
        "name": "7. Heart Disease Prediction",
        "desc": "Classification model predicting heart disease presence from clinical data.",
        "status": "Coming Soon",
        "page": None,
        "icon": "🫀",
    },
    {
        "name": "8. Face Detection",
        "desc": "Computer vision model detecting faces within images.",
        "status": "Coming Soon",
        "page": None,
        "icon": "🙂",
    },
    {
        "name": "9. Image Classification",
        "desc": "Deep learning model classifying images into multiple categories.",
        "status": "Coming Soon",
        "page": None,
        "icon": "🖼️",
    },
    {
        "name": "10. Salary Prediction",
        "desc": "Regression model estimating salary based on experience and job-related features.",
        "status": "Coming Soon",
        "page": None,
        "icon": "💰",
    },
    {
        "name": "11. Handwritten Digit Recognizer",
        "desc": "Neural network classifying hand-drawn digits in real time, trained on the MNIST dataset.",
        "status": "Live",
        "page": "pages/11_Handwritten_Digit_Recognizer.py",
        "icon": "✍️",
    },
]

# ---------- Projects Overview Section ----------
st.subheader("📁 Project Index")

for project in PROJECTS:
    is_live = project["status"] == "Live"
    status_badge = "✅ Live" if is_live else "🔜 Coming Soon"

    col_title, col_status = st.columns([3, 1])

    with col_title:
        if is_live and project["page"]:
            try:
                st.page_link(project["page"], label=f"**{project['name']}**", icon=project["icon"])
            except Exception:
                st.markdown(f"**{project['name']}** *(see sidebar)*")
        else:
            st.markdown(f"**{project['name']}**")
        st.caption(project["desc"])

    with col_status:
        st.write(f"`{status_badge}`")

    st.write("")

# ---------- Footer ----------
st.divider()

col_footer_left, col_footer_right = st.columns([2, 1])

with col_footer_left:
    st.markdown("**Tech Stack:** Python • Streamlit • Scikit-learn • Pandas • NumPy")

with col_footer_right:
    # Replace URL with your actual GitHub repository link
    st.markdown("[View on GitHub](https://github.com/OjasPal/ML-DL-Portfolio)")
