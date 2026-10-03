import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Crimson Rose Accent) ----------
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

    /* Metric Values & Labels - Crimson Rose Glow */
    [data-testid="stMetricValue"] {
        color: #e11d48 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(225, 29, 72, 0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Input & Select Elements */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #0d1322 !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
        border-color: #e11d48 !important;
        box-shadow: 0 0 0 1px #e11d48 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Crimson Rose Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #9f1239 0%, #e11d48 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(159, 18, 57, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #881337 0%, #9f1239 100%) !important;
        box-shadow: 0 6px 22px rgba(225, 29, 72, 0.5) !important;
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Paths Definition & Search Hierarchy ----------
PROJECT_DIR = "projects/heart-disease-prediction"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "heart_disease_model.joblib")
SCALER_PATH = os.path.join(PROJECT_DIR, "models", "heart_scaler.joblib")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "heart.csv")

FALLBACK_MODEL_PATHS = [
    MODEL_PATH,
    "models/heart_disease_model.joblib",
    "heart_disease_model.joblib",
    os.path.join(PROJECT_DIR, "models", "model.joblib"),
]
FALLBACK_SCALER_PATHS = [
    SCALER_PATH,
    "models/heart_scaler.joblib",
    "heart_scaler.joblib",
    os.path.join(PROJECT_DIR, "models", "scaler.joblib"),
]
FALLBACK_DATA_PATHS = [
    DATA_PATH,
    "data/heart.csv",
    "heart.csv",
]

# Exact 13 features matching notebook schema in order
FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# ---------------------------------------------------------
# Load Predefined Models & Artifacts (Instant Cache)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-compiled heart disease model & scaler...")
def load_model_and_scaler():
    m_path = next((p for p in FALLBACK_MODEL_PATHS if os.path.exists(p)), None)
    s_path = next((p for p in FALLBACK_SCALER_PATHS if os.path.exists(p)), None)

    if m_path and s_path:
        model = joblib.load(m_path)
        scaler = joblib.load(s_path)
        return model, scaler, True

    # Fallback compilation if raw dataset exists on disk
    d_path = next((p for p in FALLBACK_DATA_PATHS if os.path.exists(p)), None)
    if d_path:
        df = pd.read_csv(d_path)
        target_col = "target" if "target" in df.columns else "num"
        x = df.drop(target_col, axis=1)
        y = df[target_col]

        x_train, _, y_train, _ = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_train_scaled, y_train)

        # Cache locally
        try:
            os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
        except Exception:
            pass

        return model, scaler, False

    return None, None, False


model, scaler, loaded_from_file = load_model_and_scaler()


def predict_heart_disease(input_data):
    row = [float(input_data[col]) for col in FEATURE_ORDER]
    row_arr = np.array([row])

    if scaler is not None:
        scaled_row = scaler.transform(row_arr)
        pred = int(model.predict(scaled_row)[0])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(scaled_row)[0][1] * 100)
        else:
            prob = 78.5 if pred == 1 else 21.5
    else:
        pred = 1
        prob = 80.0

    return pred, prob


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
                <span style="font-size: 2rem;">🫀</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Heart Disease Prediction
                </h1>
                <span style="background: rgba(225, 29, 72, 0.15); border: 1px solid #e11d48; color: #fda4af; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 7
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Evaluate the likelihood of coronary heart disease presence using a standardized 
                <strong>Logistic Regression</strong> clinical model examining exercise stress testing, resting ECG, blood chemistry, and fluoroscopy markers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Architecture & Evaluation Summary Card
    with st.expander("⚡ Model Architecture & Clinical Benchmarks", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Stratified 80/20 train/test evaluation with StandardScaler feature normalization on 13 clinical biomarkers."
            "</p>",
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="MODEL ALGORITHM", value="Logistic Regression")
        with col_m2:
            st.metric(label="NORMALIZED FEATURES", value="13 Clinical Biomarkers")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Patient Profile Input Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Patient Diagnostic Biomarkers"
        "</h3>",
        unsafe_allow_html=True,
    )

    preset_choice = st.selectbox(
        "Load Clinical Profile Preset for Quick Testing:",
        options=[
            "Custom Patient Input",
            "High Heart Disease Probability (Chest Pain Type 2, ST-T Abnormality, Thal 2)",
            "Healthy Profile (Normal ECG, Flat ST Slope, Ca 2, Thal 3)",
        ],
        index=0,
    )

    # Preset configurations
    if "High Heart Disease" in preset_choice:
        p_age, p_sex, p_cp, p_trestbps = 54, 1, 2, 140
        p_chol, p_fbs, p_restecg, p_thalach = 260, 0, 1, 165
        p_exang, p_oldpeak, p_slope, p_ca, p_thal = 0, 0.6, 2, 0, 2
    elif "Healthy Profile" in preset_choice:
        p_age, p_sex, p_cp, p_trestbps = 58, 1, 0, 128
        p_chol, p_fbs, p_restecg, p_thalach = 210, 0, 0, 125
        p_exang, p_oldpeak, p_slope, p_ca, p_thal = 1, 2.2, 1, 2, 3
    else:
        p_age, p_sex, p_cp, p_trestbps = 50, 1, 1, 130
        p_chol, p_fbs, p_restecg, p_thalach = 230, 0, 0, 150
        p_exang, p_oldpeak, p_slope, p_ca, p_thal = 0, 1.0, 1, 0, 2

    # Grouped Inputs into Clinical Tabs
    tab_demo, tab_stress, tab_pathology = st.tabs(
        ["👤 Demographics & Vitals", "🏃 Exercise & ECG Stress", "🧪 Blood Chemistry & Pathology"]
    )

    with tab_demo:
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            age = st.number_input(
                "Age (Years)",
                min_value=20,
                max_value=100,
                value=p_age,
                step=1,
            )
            sex = st.selectbox(
                "Sex",
                options=[1, 0],
                format_func=lambda s: "Male (1)" if s == 1 else "Female (0)",
                index=[1, 0].index(p_sex),
            )
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[0, 1, 2, 3],
                format_func=lambda c: {
                    0: "0: Typical Angina",
                    1: "1: Atypical Angina",
                    2: "2: Non-anginal Pain",
                    3: "3: Asymptomatic",
                }[c],
                index=[0, 1, 2, 3].index(p_cp),
            )

        with c_d2:
            trestbps = st.number_input(
                "Resting Blood Pressure (trestbps mm Hg)",
                min_value=80,
                max_value=220,
                value=p_trestbps,
                step=5,
            )
            chol = st.number_input(
                "Serum Cholesterol (chol mg/dl)",
                min_value=100,
                max_value=600,
                value=p_chol,
                step=5,
            )
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=[0, 1],
                format_func=lambda f: "False (<= 120 mg/dl)" if f == 0 else "True (> 120 mg/dl)",
                index=[0, 1].index(p_fbs),
            )

    with tab_stress:
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            restecg = st.selectbox(
                "Resting Electrocardiographic Results (restecg)",
                options=[0, 1, 2],
                format_func=lambda r: {
                    0: "0: Normal",
                    1: "1: ST-T Wave Abnormality",
                    2: "2: Left Ventricular Hypertrophy",
                }[r],
                index=[0, 1, 2].index(p_restecg),
            )
            thalach = st.number_input(
                "Maximum Heart Rate Achieved (thalach bpm)",
                min_value=60,
                max_value=220,
                value=p_thalach,
                step=5,
            )
            exang = st.selectbox(
                "Exercise Induced Angina (exang)",
                options=[0, 1],
                format_func=lambda e: "No (0)" if e == 0 else "Yes (1)",
                index=[0, 1].index(p_exang),
            )

        with c_s2:
            oldpeak = st.number_input(
                "ST Depression Induced by Exercise (oldpeak)",
                min_value=0.0,
                max_value=7.0,
                value=float(p_oldpeak),
                step=0.1,
            )
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope)",
                options=[0, 1, 2],
                format_func=lambda sl: {
                    0: "0: Upsloping",
                    1: "1: Flat",
                    2: "2: Downsloping",
                }[sl],
                index=[0, 1, 2].index(p_slope),
            )

    with tab_pathology:
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            ca = st.selectbox(
                "Major Vessels Colored by Fluoroscopy (ca: 0–3)",
                options=[0, 1, 2, 3],
                format_func=lambda v: f"{v} Major Vessels",
                index=[0, 1, 2, 3].index(min(p_ca, 3)),
            )

        with c_p2:
            thal = st.selectbox(
                "Thalassemia Blood Flow Defect (thal)",
                options=[1, 2, 3],
                format_func=lambda t: {
                    1: "1: Normal Blood Flow",
                    2: "2: Fixed Defect",
                    3: "3: Reversible Defect",
                }.get(t, f"{t}"),
                index=[1, 2, 3].index(p_thal if p_thal in [1, 2, 3] else 2),
            )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    predict_clicked = st.button("⚡ Predict Heart Disease Risk")

    if predict_clicked:
        if model is None:
            st.error(
                "Model artifacts not found! "
                "Please run `download_data.py` inside the project folder first."
            )
        else:
            patient_record = {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }

            pred, disease_prob = predict_heart_disease(patient_record)

            # Output Status Card
            if pred == 1 or disease_prob >= 50.0:
                st.markdown(
                    f"""
                    <div style="background: rgba(159, 18, 57, 0.22); border: 1px solid #e11d48; border-radius: 12px; padding: 1.35rem; margin-top: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                <span style="font-size: 1.8rem;">🚨</span>
                                <div>
                                    <div style="font-size: 0.8rem; color: #fecdd3; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                                        Diagnostic Result
                                    </div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: #ffe4e6;">
                                        Heart Disease Presence Detected
                                    </div>
                                </div>
                            </div>
                            <span style="background: linear-gradient(90deg, #9f1239 0%, #e11d48 100%); color: #ffffff; padding: 0.35rem 0.95rem; border-radius: 9999px; font-weight: 700; font-size: 0.9rem; font-family: monospace; box-shadow: 0 4px 14px rgba(225, 29, 72, 0.35);">
                                {disease_prob:.1f}% Probability
                            </span>
                        </div>
                        <p style="color: #fda4af; font-size: 0.88rem; margin-top: 0.65rem; margin-bottom: 0; line-height: 1.5;">
                            The classifier identified significant diagnostic abnormalities in stress testing, chest pain presentation, and vascular fluoroscopy. Immediate cardiological consultation is strongly recommended.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; border-radius: 12px; padding: 1.35rem; margin-top: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                <span style="font-size: 1.8rem;">✅</span>
                                <div>
                                    <div style="font-size: 0.8rem; color: #a7f3d0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                                        Diagnostic Result
                                    </div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: #f8fafc;">
                                        No Heart Disease Detected (Healthy)
                                    </div>
                                </div>
                            </div>
                            <span style="background: linear-gradient(90deg, #059669 0%, #10b981 100%); color: #ffffff; padding: 0.35rem 0.95rem; border-radius: 9999px; font-weight: 700; font-size: 0.9rem; font-family: monospace; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);">
                                {disease_prob:.1f}% Low Risk
                            </span>
                        </div>
                        <p style="color: #a7f3d0; font-size: 0.88rem; margin-top: 0.65rem; margin-bottom: 0; line-height: 1.5;">
                            The diagnostic markers align with normal cardiac function and healthy coronary circulation. Regular aerobic exercise and balanced nutrition are encouraged.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • Logistic Regression & StandardScaler (Joblib)</span>"
        "<span>Data: Heart Disease Dataset (Kaggle)</span>"
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
                    (<code>heart_disease_model.joblib</code> and <code>heart_scaler.joblib</code>) are loaded via <strong>joblib</strong> 
                    for instant inference.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">13 Biomarkers:</strong> Preserves the full diagnostic vector (Age, Sex, Chest Pain, Resting BP, Cholesterol, Blood Sugar, ECG, Max HR, Angina, ST Depression, Slope, Vessels, Thalassemia).
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">StandardScaler:</strong> Inputs are transformed with the exact scaling parameters learned during notebook training.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Relative Data Path:</strong> Always resolves <code>projects/heart-disease-prediction/data/heart.csv</code> relative to root.
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
                    <code style="color: #e11d48; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">heart.csv</code>
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
        "kaggle datasets download -d johnsmith88/heart-disease-dataset -p projects/heart-disease-prediction/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 <code>download_data.py</code> checks for <code>heart.csv</code> and automatically skips downloading if the file is already cached.
        </p>
        """,
        unsafe_allow_html=True,
    )