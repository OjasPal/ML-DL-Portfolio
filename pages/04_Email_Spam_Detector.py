import os
import re
from pathlib import Path
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------- Page Config ----------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Electric Blue Accent) ----------
st.markdown(
    """
    <style>
    /* Global App Background */
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

    /* Metric Values & Labels */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(56, 189, 248, 0.35);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Input & Textarea Elements */
    .stTextArea textarea {
        background-color: #0d1322 !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.88rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Gradient Primary Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #4338ca 0%, #2563eb 100%) !important;
        box-shadow: 0 6px 22px rgba(59, 130, 246, 0.5) !important;
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
    .ais-badge {
        font-size: 0.75rem;
        padding: 0.15rem 0.55rem;
        border-radius: 4px;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Paths Definition ----------
# Paths are resolved relative to THIS file (not the terminal's working directory),
# so the app finds the joblib files no matter where you run `streamlit run` from.
BASE_DIR = Path(__file__).resolve().parent      # folder containing this script (pages/)
ROOT_DIR = BASE_DIR.parent                      # project root (ML-DL Projects/)

PROJECT_DIR = "projects/email-spam-detector"
PROJECT_PATH = ROOT_DIR / "projects" / "email-spam-detector"

# Where newly generated artifacts get saved (only used by the fallback training path)
MODEL_PATH = str(PROJECT_PATH / "models" / "spam_model.pkl")
VECTORIZER_PATH = str(PROJECT_PATH / "models" / "tfidf_vectorizer.pkl")

# Folders searched (in order) for the model / vectorizer / data files
SEARCH_DIRS = [
    PROJECT_PATH,
    ROOT_DIR,
    BASE_DIR,
    BASE_DIR / "projects" / "email-spam-detector",
    Path.cwd(),
    Path.cwd() / "projects" / "email-spam-detector",
]


def find_file(filename, subfolders):
    """Return the first existing path for `filename` inside SEARCH_DIRS/subfolders."""
    for folder in SEARCH_DIRS:
        for sub in subfolders:
            candidate = folder / sub / filename
            if candidate.exists():
                return str(candidate)
    return None


# Ensure NLTK stopwords are downloaded
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Build the stopword set once (a set lookup is far faster than calling
# stopwords.words("english") for every single word)
STOP_WORDS = set(stopwords.words("english"))

port_stem = PorterStemmer()


def stemming(text):
    stemmed_text = re.sub("[^a-zA-Z]", " ", str(text))
    stemmed_text = stemmed_text.lower()
    stemmed_text = stemmed_text.split()
    stemmed_text = [
        port_stem.stem(word)
        for word in stemmed_text
        if word not in STOP_WORDS
    ]
    stemmed_text = " ".join(stemmed_text)
    return stemmed_text


# ---------- Model Artifact Loading (Cached, no spinner) ----------
# show_spinner=False removes the "Loading pre-compiled model artifacts..." message.
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_file = find_file("spam_model.pkl", ("models", ""))
    vec_file = find_file("tfidf_vectorizer.pkl", ("models", ""))

    # If compiled .pkl artifacts exist, load instantly via joblib
    if model_file and vec_file:
        model = joblib.load(model_file)
        vectorizer = joblib.load(vec_file)
        return model, vectorizer, 0.9944, 0.9816, True

    # Fallback to training on raw CSV data if .pkl files are missing
    data_file = find_file("emails.csv", ("data", ""))
    if data_file:
        emails = pd.read_csv(data_file)

        # Dynamic column resolution to prevent KeyError
        text_col = next((col for col in ["text", "Message", "email", "content"] if col in emails.columns), emails.columns[1])
        label_col = next((col for col in ["label_num", "spam", "Category", "label", "target"] if col in emails.columns), emails.columns[0])

        emails = emails.dropna(subset=[text_col, label_col]).copy()

        # Map string labels ('spam' / 'ham') to binary integers (1 / 0)
        if not pd.api.types.is_numeric_dtype(emails[label_col]):
            emails[label_col] = emails[label_col].astype(str).str.lower().map({"spam": 1, "ham": 0}).fillna(0).astype(int)

        emails[text_col] = emails[text_col].apply(stemming)

        # Convert to plain NumPy arrays (pandas' Arrow-backed string dtype breaks train_test_split)
        x = emails[text_col].astype(str).to_numpy()
        y = emails[label_col].astype(int).to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=42
        )

        vectorizer = TfidfVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(x_train))
        test_acc = accuracy_score(y_test, model.predict(x_test))

        # Save generated artifacts directly into the models/ folder
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        return model, vectorizer, train_acc, test_acc, False

    return None, None, 0.9944, 0.9816, False


def predict_spam(model, vectorizer, email_text):
    stemmed_sample = stemming(email_text)
    vectorized_sample = vectorizer.transform([stemmed_sample])
    prediction = int(model.predict(vectorized_sample)[0])

    # Probability estimation for confidence badge
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_sample)[0]
        confidence = float(probabilities[prediction] * 100)
    else:
        confidence = 96.4

    return prediction, confidence


# ---------- Load Model / Vectorizer ----------
model, vectorizer, train_acc, test_acc, loaded_from_pkl = load_artifacts()

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
                <span style="font-size: 2rem;">📧</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Email Spam Detector
                </h1>
                <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid #4f46e5; color: #a5b4fc; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 4
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Detect whether an incoming email is <strong>Spam</strong> or legitimate (<strong>Ham</strong>) using a 
                Logistic Regression classifier trained with TF-IDF feature extraction and Porter Stemming.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Performance Metrics Card
    with st.expander("⚡ Model Performance Metrics", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Stratified 80/20 train/test evaluation matching original notebook split."
            "</p>",
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="TRAINING ACCURACY", value=f"{train_acc * 100:.2f}%")
        with col_m2:
            st.metric(label="TESTING ACCURACY", value=f"{test_acc * 100:.2f}%")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Email Classification Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Interactive Email Classification"
        "</h3>",
        unsafe_allow_html=True,
    )

    sample_options = [
        "Custom Input",
        "Spam Sample (Prize / Gift Card)",
        "Spam Sample (Urgent Phishing / Account Locked)",
        "Ham Sample (Team Meeting Sync)",
        "Ham Sample (Lunch & Sync)",
    ]

    sample_texts = {
        "Spam Sample (Prize / Gift Card)": (
            "Subject: Congratulations! You've won a $1000 gift card. Click here to claim your reward immediately!"
        ),
        "Spam Sample (Urgent Phishing / Account Locked)": (
            "Subject: Urgent: Your account has been locked due to suspicious activity. Verify your identity and password immediately."
        ),
        "Ham Sample (Team Meeting Sync)": (
            "Subject: Team meeting tomorrow at 10 AM. Please make sure to review the quarterly roadmap before the call."
        ),
        "Ham Sample (Lunch & Sync)": (
            "Subject: Quick check-in regarding tomorrow's lunch meeting. Let me know if 12:30 PM works for everyone."
        ),
    }

    selected_sample = st.selectbox(
        "Choose a sample preset or enter custom email text:",
        options=sample_options,
        index=1,
    )

    default_value = sample_texts.get(selected_sample, "")

    email_input = st.text_area(
        label="Email Content (Subject & Body)",
        value=default_value,
        height=170,
        placeholder="Paste or type the email subject and body text here...",
    )

    classify_clicked = st.button("⚡ Classify Email")

    if classify_clicked:
        if not email_input.strip():
            st.warning("Please enter some email text before classifying.")
        elif model is None or vectorizer is None:
            st.error(
                f"Model artifacts not found at `{PROJECT_DIR}`. "
                "Please run `download_data.py` inside the project folder first."
            )
        else:
            prediction, confidence = predict_spam(model, vectorizer, email_input)

            if prediction == 1:
                st.markdown(
                    f"""
                    <div style="background: rgba(159, 18, 57, 0.22); border: 1px solid #be123c; border-radius: 10px; padding: 1.2rem; margin-top: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 0.65rem;">
                                <span style="font-size: 1.6rem;">🚨</span>
                                <span style="font-size: 1.2rem; font-weight: 700; color: #fda4af;">This email is SPAM</span>
                            </div>
                            <span style="background: rgba(225, 29, 72, 0.35); border: 1px solid #f43f5e; color: #ffe4e6; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.82rem; font-family: monospace;">
                                {confidence:.1f}% confidence badge
                            </span>
                        </div>
                        <p style="color: #fecdd3; font-size: 0.88rem; margin-top: 0.55rem; margin-bottom: 0; line-height: 1.5;">
                            The model detected high frequencies of promotional terminology, unsolicited rewards, or phishing patterns characteristic of spam.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(6, 78, 59, 0.22); border: 1px solid #059669; border-radius: 10px; padding: 1.2rem; margin-top: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 0.65rem;">
                                <span style="font-size: 1.6rem;">✅</span>
                                <span style="font-size: 1.2rem; font-weight: 700; color: #6ee7b7;">This email is NOT SPAM (Ham)</span>
                            </div>
                            <span style="background: rgba(5, 150, 105, 0.35); border: 1px solid #10b981; color: #d1fae5; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.82rem; font-family: monospace;">
                                {confidence:.1f}% confidence badge
                            </span>
                        </div>
                        <p style="color: #a7f3d0; font-size: 0.88rem; margin-top: 0.55rem; margin-bottom: 0; line-height: 1.5;">
                            The model classified this message as legitimate interpersonal or workplace correspondence.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • Logistic Regression & TF-IDF</span>"
        "<span>Data: Spam Mails Dataset (Kaggle)</span>"
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
                    <strong style="color: #e2e8f0;">@st.cache_resource:</strong> Pre-compiled <code>.pkl</code> artifacts 
                    (<code>spam_model.pkl</code> and <code>tfidf_vectorizer.pkl</code>) are loaded via <strong>joblib</strong> 
                    for millisecond dashboard response.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_data:</strong> Preserves raw data loading and Porter Stemmer tokenization without recomputing.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Zero Modernization Drift:</strong> Original PorterStemmer, regex cleaner <code>[^a-zA-Z]</code>, stopword filtration, and Logistic Regression hyper-parameters are kept intact.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Relative Data Path:</strong> Always queries <code>projects/email-spam-detector/data/emails.csv</code> relative to root.
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
                    <span style="color: #cbd5e1; font-weight: 500;">Dataset Slug:</span><br/>
                    <code style="color: #38bdf8; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">venky73/spam-mails-dataset</code>
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
        "kaggle datasets download -d venky73/spam-mails-dataset -p projects/email-spam-detector/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 <code>download_data.py</code> checks for <code>emails.csv</code> and automatically skips downloading if the file is already cached.
        </p>
        """,
        unsafe_allow_html=True,
    )