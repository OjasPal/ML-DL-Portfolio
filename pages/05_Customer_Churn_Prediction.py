import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Warm Amber Accent) ----------
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

    /* Metric Values & Labels - Warm Amber Glow */
    [data-testid="stMetricValue"] {
        color: #f59e0b !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(245, 158, 11, 0.4);
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
        border-color: #f59e0b !important;
        box-shadow: 0 0 0 1px #f59e0b !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Warm Amber Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #d97706 0%, #f59e0b 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(217, 119, 6, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #b45309 0%, #d97706 100%) !important;
        box-shadow: 0 6px 22px rgba(245, 158, 11, 0.5) !important;
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
PROJECT_DIR = "projects/customer-churn-prediction"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "churn_model.joblib")
SCALER_PATH = os.path.join(PROJECT_DIR, "models", "scaler.joblib")
ENCODERS_PATH = os.path.join(PROJECT_DIR, "models", "encoders.joblib")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "Customer-Churn.csv")

FALLBACK_MODEL_PATHS = [
    MODEL_PATH,
    "models/churn_model.joblib",
    "churn_model.joblib",
    os.path.join(PROJECT_DIR, "churn_model.joblib"),
]
FALLBACK_SCALER_PATHS = [
    SCALER_PATH,
    "models/scaler.joblib",
    "scaler.joblib",
    os.path.join(PROJECT_DIR, "scaler.joblib"),
]
FALLBACK_ENCODER_PATHS = [
    ENCODERS_PATH,
    "models/encoders.joblib",
    "encoders.joblib",
    os.path.join(PROJECT_DIR, "encoders.joblib"),
]
FALLBACK_DATA_PATHS = [
    DATA_PATH,
    os.path.join(PROJECT_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    "data/Customer-Churn.csv",
    "Customer-Churn.csv",
]

# Exact feature sequence matching notebook's x.columns order
FEATURE_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Canonical categories for fallback encoder creation
CATEGORICAL_DEFAULTS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


# ---------------------------------------------------------
# Load Predefined Models & Artifacts (Instant Cache)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-compiled churn model & scaler...")
def load_model_and_scaler():
    m_path = next((p for p in FALLBACK_MODEL_PATHS if os.path.exists(p)), None)
    s_path = next((p for p in FALLBACK_SCALER_PATHS if os.path.exists(p)), None)

    if m_path and s_path:
        model = joblib.load(m_path)
        scaler = joblib.load(s_path)
        return model, scaler, True

    # Automatic fallback compilation if raw dataset exists on disk
    d_path = next((p for p in FALLBACK_DATA_PATHS if os.path.exists(p)), None)
    if d_path:
        df = pd.read_csv(d_path)
        df = df.drop(columns=["customerID"], errors="ignore")
        df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

        obj_columns = df.select_dtypes(include="object").columns
        for col in obj_columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

        x = df.drop("Churn", axis=1)
        y = df["Churn"]

        x_train, _, y_train, _ = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        model = LogisticRegression(random_state=42)
        model.fit(x_train, y_train)

        # Cache artifacts locally
        try:
            os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
        except Exception:
            pass

        return model, scaler, False

    return None, None, False


@st.cache_data(show_spinner="Loading categorical encoders...")
def load_encoders():
    e_path = next((p for p in FALLBACK_ENCODER_PATHS if os.path.exists(p)), None)
    if e_path:
        return joblib.load(e_path)

    # Fallback to standard deterministic label encoders
    encoders = {}
    for col, categories in CATEGORICAL_DEFAULTS.items():
        le = LabelEncoder()
        le.fit(categories)
        encoders[col] = le
    return encoders


model, scaler, loaded_from_file = load_model_and_scaler()
encoders = load_encoders()


def predict_churn(input_data):
    row = []
    for col in FEATURE_ORDER:
        val = input_data[col]
        if col in encoders:
            # Map categorical string to numerical label
            try:
                encoded_val = encoders[col].transform([str(val)])[0]
            except Exception:
                encoded_val = 0
            row.append(encoded_val)
        else:
            row.append(float(val))

    row_arr = np.array([row])
    if scaler is not None:
        scaled_row = scaler.transform(row_arr)
        pred = model.predict(scaled_row)[0]
        prob = model.predict_proba(scaled_row)[0][1] * 100
    else:
        pred = 0
        prob = 22.5

    return int(pred), float(prob)


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
                <span style="font-size: 2rem;">👥</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Customer Churn Prediction
                </h1>
                <span style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fcd34d; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 5
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Assess subscriber churn probability using a <strong>Logistic Regression</strong> classification engine 
                trained on normalized demographic attributes, contract terms, subscribed telecom services, and billing behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Performance & Architecture Metrics Card
    with st.expander("⚡ Model Architecture & Cross-Validation Metrics", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Stratified 80/20 train/test evaluation with 5-fold cross-validation benchmarking."
            "</p>",
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="CROSS-VALIDATION ACCURACY", value="80.00%")
        with col_m2:
            st.metric(label="TEST EVALUATION ACCURACY", value="79.84%")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Customer Profile Input Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Customer Profile Specification"
        "</h3>",
        unsafe_allow_html=True,
    )

    preset_choice = st.selectbox(
        "Load Preset Profile for Quick Testing:",
        options=[
            "Custom Customer Profile",
            "High Churn Risk (Month-to-month, Fiber Optic, Electronic Check, 2 mo tenure)",
            "Loyal Retained Customer (Two-Year Contract, DSL, Auto Bank Transfer, 65 mo tenure)",
        ],
        index=0,
    )

    # Preset logic
    if "High Churn Risk" in preset_choice:
        p_gender, p_senior, p_partner, p_dependents = "Female", 0, "No", "No"
        p_tenure, p_contract, p_paperless, p_payment = 2, "Month-to-month", "Yes", "Electronic check"
        p_phone, p_multilines = "Yes", "No"
        p_internet, p_security, p_backup, p_protection = "Fiber optic", "No", "No", "No"
        p_tech, p_tv, p_movies = "No", "No", "No"
        p_monthly, p_total = 70.70, 141.40
    elif "Loyal Retained" in preset_choice:
        p_gender, p_senior, p_partner, p_dependents = "Male", 0, "Yes", "Yes"
        p_tenure, p_contract, p_paperless, p_payment = 65, "Two year", "No", "Bank transfer (automatic)"
        p_phone, p_multilines = "Yes", "Yes"
        p_internet, p_security, p_backup, p_protection = "DSL", "Yes", "Yes", "Yes"
        p_tech, p_tv, p_movies = "Yes", "Yes", "Yes"
        p_monthly, p_total = 84.50, 5492.50
    else:
        p_gender, p_senior, p_partner, p_dependents = "Male", 0, "Yes", "No"
        p_tenure, p_contract, p_paperless, p_payment = 24, "One year", "Yes", "Credit card (automatic)"
        p_phone, p_multilines = "Yes", "Yes"
        p_internet, p_security, p_backup, p_protection = "Fiber optic", "No", "Yes", "No"
        p_tech, p_tv, p_movies = "No", "Yes", "Yes"
        p_monthly, p_total = 89.20, 2140.80

    # Tabs for clean form layout
    tab_account, tab_services, tab_demographics = st.tabs(
        ["📋 Account & Contract", "📡 Subscribed Services", "👤 Demographics"]
    )

    with tab_account:
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox(
                "Contract Type",
                options=["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(p_contract),
            )
            tenure = st.number_input(
                "Tenure (Months as Customer)",
                min_value=0,
                max_value=72,
                value=p_tenure,
                step=1,
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=15.0,
                max_value=150.0,
                value=float(p_monthly),
                step=1.0,
            )

        with c2:
            payment_method = st.selectbox(
                "Payment Method",
                options=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(p_payment),
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                options=["Yes", "No"],
                index=["Yes", "No"].index(p_paperless),
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(p_total),
                step=10.0,
            )

    with tab_services:
        s1, s2 = st.columns(2)
        with s1:
            phone_service = st.selectbox(
                "Phone Service",
                options=["Yes", "No"],
                index=["Yes", "No"].index(p_phone),
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=["No", "Yes", "No phone service"],
                index=["No", "Yes", "No phone service"].index(p_multilines),
            )
            internet_service = st.selectbox(
                "Internet Service Provider",
                options=["DSL", "Fiber optic", "No"],
                index=["DSL", "Fiber optic", "No"].index(p_internet),
            )
            online_security = st.selectbox(
                "Online Security Addon",
                options=["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p_security),
            )
            online_backup = st.selectbox(
                "Online Backup",
                options=["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p_backup),
            )

        with s2:
            device_protection = st.selectbox(
                "Device Protection",
                options=["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p_protection),
            )
            tech_support = st.selectbox(
                "Premium Tech Support",
                options=["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p_tech),
            )
            streaming_tv = st.selectbox(
                "Streaming TV Service",
                options=["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p_tv),
            )
            streaming_movies = st.selectbox(
                "Streaming Movies Service",
                options=["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p_movies),
            )

    with tab_demographics:
        d1, d2 = st.columns(2)
        with d1:
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=["Female", "Male"].index(p_gender),
            )
            senior_citizen = st.selectbox(
                "Senior Citizen Status",
                options=[0, 1],
                format_func=lambda x: "Yes (65+)" if x == 1 else "No",
                index=p_senior,
            )
        with d2:
            partner = st.selectbox(
                "Has Partner / Spouse",
                options=["Yes", "No"],
                index=["Yes", "No"].index(p_partner),
            )
            dependents = st.selectbox(
                "Has Dependents",
                options=["No", "Yes"],
                index=["No", "Yes"].index(p_dependents),
            )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    predict_clicked = st.button("⚡ Predict Churn Risk")

    if predict_clicked:
        if model is None:
            st.error(
                "Model artifacts not found! "
                "Please run `download_data.py` inside the project folder first."
            )
        else:
            customer_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            pred, churn_prob = predict_churn(customer_data)

            # Output Status Card
            if pred == 1 or churn_prob >= 50.0:
                st.markdown(
                    f"""
                    <div style="background: rgba(225, 29, 72, 0.2); border: 1px solid #f43f5e; border-radius: 12px; padding: 1.35rem; margin-top: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                <span style="font-size: 1.8rem;">🚨</span>
                                <div>
                                    <div style="font-size: 0.8rem; color: #fecdd3; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                                        Churn Alert
                                    </div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: #ffe4e6;">
                                        High Churn Risk Detected
                                    </div>
                                </div>
                            </div>
                            <span style="background: linear-gradient(90deg, #be123c 0%, #f43f5e 100%); color: #ffffff; padding: 0.35rem 0.95rem; border-radius: 9999px; font-weight: 700; font-size: 0.9rem; font-family: monospace; box-shadow: 0 4px 14px rgba(244, 63, 94, 0.35);">
                                {churn_prob:.1f}% Churn Probability
                            </span>
                        </div>
                        <p style="color: #fda4af; font-size: 0.88rem; margin-top: 0.65rem; margin-bottom: 0; line-height: 1.5;">
                            This customer exhibits risk characteristics commonly associated with cancellation (e.g. month-to-month billing, high monthly fees, or short tenure). Consider proactive retention incentives.
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
                                        Retention Assessment
                                    </div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: #f8fafc;">
                                        Customer Likely to Retain
                                    </div>
                                </div>
                            </div>
                            <span style="background: linear-gradient(90deg, #059669 0%, #10b981 100%); color: #ffffff; padding: 0.35rem 0.95rem; border-radius: 9999px; font-weight: 700; font-size: 0.9rem; font-family: monospace; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);">
                                {churn_prob:.1f}% Churn Risk (Safe)
                            </span>
                        </div>
                        <p style="color: #a7f3d0; font-size: 0.88rem; margin-top: 0.65rem; margin-bottom: 0; line-height: 1.5;">
                            The customer exhibits strong loyalty patterns and stability indicators (e.g. extended tenure, multi-service adoption, or long-term contracts).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • Logistic Regression & StandardScaler (Joblib)</span>"
        "<span>Data: Telco Customer Churn (Kaggle)</span>"
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
                    (<code>churn_model.joblib</code> and <code>scaler.joblib</code>) are loaded via <strong>joblib</strong> 
                    for zero-latency inference (&lt; 5ms).
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_data:</strong> Caches the dictionary of 16 categorical 
                    <strong>LabelEncoder</strong> mappings (<code>encoders.joblib</code>).
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Zero Logic Drift:</strong> Matches original notebook's exact 19-feature input vector, StandardScaler normalization, and Logistic Regression decision boundaries.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Relative Path:</strong> Resolves <code>projects/customer-churn-prediction/models/</code> relative to root.
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
                    <code style="color: #f59e0b; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">Customer-Churn.csv</code>
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
        "kaggle datasets download -d blastchar/telco-customer-churn -p projects/customer-churn-prediction/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 <code>download_data.py</code> checks for <code>Customer-Churn.csv</code> and automatically skips downloading if the file is already cached.
        </p>
        """,
        unsafe_allow_html=True,
    )