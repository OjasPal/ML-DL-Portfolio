import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---------- Page Config ----------
st.set_page_config(
    page_title="Delhi Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Emerald Green Accent) ----------
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

    /* Metric Values & Labels - Emerald Green Glow */
    [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(16, 185, 129, 0.4);
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
        border-color: #10b981 !important;
        box-shadow: 0 0 0 1px #10b981 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Emerald Green Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%) !important;
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
        background: linear-gradient(90deg, #047857 0%, #059669 100%) !important;
        box-shadow: 0 6px 22px rgba(16, 185, 129, 0.5) !important;
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

# ---------- Paths Definition ----------
PROJECT_DIR = "projects/housing-price-prediction"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "Delhi.csv")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "housing_price_model.joblib")
SCALER_PATH = os.path.join(PROJECT_DIR, "models", "housing_scaler.joblib")

FALLBACK_DATA_PATHS = [
    DATA_PATH,
    "data/Delhi.csv",
    "Delhi.csv",
    os.path.join(PROJECT_DIR, "Delhi.csv"),
]
FALLBACK_MODEL_PATHS = [
    MODEL_PATH,
    "models/housing_price_model.joblib",
    "housing_price_model.joblib",
    os.path.join(PROJECT_DIR, "housing_price_model.joblib"),
]
FALLBACK_SCALER_PATHS = [
    SCALER_PATH,
    "models/housing_scaler.joblib",
    "housing_scaler.joblib",
    os.path.join(PROJECT_DIR, "housing_scaler.joblib"),
]

# Standard default feature columns from notebook schema
DEFAULT_FEATURE_COLS = [
    "Area", "Location", "No. of Bedrooms", "Resale", "MaintenanceStaff",
    "Gymnasium", "SwimmingPool", "LandscapedGardens", "JoggingTrack",
    "RainWaterHarvesting", "IndoorGames", "ShoppingMall", "Intercom",
    "SportsFacility", "24X7Security", "PowerBackup", "CarParking",
    "StaffQuarter", "Cafeteria", "MultipurposeRoom", "Hospital",
    "WashingMachine", "Gasconnection", "AC", "Wifi", "Children'splayarea",
    "LiftAvailable", "BED", "VaastuCompliant", "Microwave", "GolfCourse",
    "TV", "DiningTable", "Sofa", "Wardrobe", "Refrigerator"
]

DEFAULT_LOCATIONS = [
    "Alaknanda", "Chhatarpur", "Commonwealth Games Village", "Connaught Place",
    "Dwarka Mor", "Dwarka Sector 10", "Dwarka Sector 11", "Dwarka Sector 12",
    "Dwarka Sector 19", "Dwarka Sector 22", "Dwarka Sector 6", "Greater Kailash",
    "Hauz Khas", "Janakpuri", "Jasola", "Karol Bagh", "Lajpat Nagar",
    "Mahavir Enclave", "Malviya Nagar", "Mayur Vihar", "Mehrauli",
    "New Friends Colony", "Paschim Vihar", "Patel Nagar", "Pitampura",
    "Punjabi Bagh", "Rohini Sector 13", "Rohini Sector 24", "Rohini Sector 9",
    "Safdarjung Enclave", "Saket", "Sarita Vihar", "South Extension",
    "Uttam Nagar", "Vasant Kunj", "Vasant Vihar"
]


# ---------- Cached Dataset & Location Metadata Loading ----------
@st.cache_data(show_spinner="Loading location metadata...")
def load_metadata():
    data_file = next((p for p in FALLBACK_DATA_PATHS if os.path.exists(p)), None)
    if data_file:
        housing = pd.read_csv(data_file)
        label_encoder = LabelEncoder()
        housing["Location_Encoded"] = label_encoder.fit_transform(housing["Location"])
        locations_list = sorted(housing["Location"].unique().tolist())

        # Extract feature columns matching exact model training order
        feature_df = housing.drop(
            ["Price", "Price_in_Millions", "Location_Encoded"],
            axis=1,
            errors="ignore",
        ).copy()
        feature_df["Location"] = housing["Location_Encoded"]
        feature_cols = list(feature_df.columns)
        return label_encoder, locations_list, feature_cols

    # Graceful fallback if dataset is not yet downloaded
    encoder = LabelEncoder()
    encoder.fit(DEFAULT_LOCATIONS)
    return encoder, DEFAULT_LOCATIONS, DEFAULT_FEATURE_COLS


# ---------- Instant Model Artifact Loading via Joblib ----------
@st.cache_resource(show_spinner="Loading pre-compiled regression artifacts...")
def load_artifacts():
    model_file = next((p for p in FALLBACK_MODEL_PATHS if os.path.exists(p)), None)
    scaler_file = next((p for p in FALLBACK_SCALER_PATHS if os.path.exists(p)), None)

    if model_file and scaler_file:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler, True

    return None, None, False


def predict_price(model, scaler, feature_cols, input_values):
    row = np.array([[input_values.get(col, 0) for col in feature_cols]])
    if scaler is not None:
        scaled_row = scaler.transform(row)
        price_millions = model.predict(scaled_row)[0]
    else:
        price_millions = model.predict(row)[0]

    # Convert millions to INR
    price_inr = price_millions * 1e6
    return max(0.0, price_inr)


# ---------- Load Artifacts ----------
label_encoder, locations_list, feature_cols = load_metadata()
model, scaler, artifacts_loaded = load_artifacts()

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
                <span style="font-size: 2rem;">🏠</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Delhi Housing Price Predictor
                </h1>
                <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 2
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Estimate real estate valuations across Metropolitan Delhi based on location, built-up area, bedroom count, 
                and comprehensive property amenities using a <strong>Linear Regression</strong> model standardized with <strong>StandardScaler</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Performance Metrics Card
    with st.expander("⚡ Model Architecture & Training Summary", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Stratified Shuffle Split on property resale condition with StandardScaler normalization."
            "</p>",
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="MODEL ALGORITHM", value="Linear Regression")
        with col_m2:
            st.metric(label="FEATURE NORMALIZATION", value="StandardScaler (Joblib)")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Property Details Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Property Specifications"
        "</h3>",
        unsafe_allow_html=True,
    )

    col_prop1, col_prop2 = st.columns(2)
    with col_prop1:
        selected_location = st.selectbox(
            "Location / Neighborhood in Delhi",
            options=locations_list,
            index=0 if "Saket" not in locations_list else locations_list.index("Saket"),
        )
        area = st.number_input(
            "Built-up Area (in sq. ft.)",
            min_value=200,
            max_value=15000,
            value=1350,
            step=50,
        )

    with col_prop2:
        bedrooms = st.selectbox(
            "Number of Bedrooms (BHK)",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            index=2,  # Default 3 BHK
        )
        resale_option = st.radio(
            "Property Status",
            options=["Resale Property", "New / Fresh Construction"],
            index=0,
            horizontal=True,
        )
        is_resale = 1 if resale_option == "Resale Property" else 0

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Amenities Section
    amenity_cols = [
        col for col in feature_cols
        if col not in ["Area", "Location", "No. of Bedrooms", "Resale"]
    ]

    amenity_values = {}
    with st.expander("✨ Select Available Amenities & Facilities", expanded=False):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Select all facilities available in the residential society to refine the price estimate."
            "</p>",
            unsafe_allow_html=True,
        )
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        for i, amenity in enumerate(amenity_cols):
            target_col = [sub_col1, sub_col2, sub_col3][i % 3]
            clean_name = (
                amenity.replace("24X7Security", "24x7 Security")
                .replace("Children'splayarea", "Children's Play Area")
                .replace("BED", "Bed Available")
                .replace("AC", "Air Conditioning")
            )
            with target_col:
                amenity_values[amenity] = (
                    1 if st.checkbox(clean_name, value=False, key=f"amenity_{amenity}") else 0
                )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    predict_clicked = st.button("⚡ Predict Price")

    if predict_clicked:
        if area <= 0:
            st.warning("Please enter a valid property area in square feet.")
        elif model is None or scaler is None:
            st.error(
                "Pre-compiled model artifacts (`housing_price_model.joblib` and `housing_scaler.joblib`) "
                "were not found in `projects/housing-price-prediction/models/`. "
                "Please ensure the exported joblib artifacts are placed in the models folder."
            )
        else:
            try:
                encoded_loc = label_encoder.transform([selected_location])[0]
            except Exception:
                encoded_loc = 0

            input_dict = {
                "Area": float(area),
                "Location": encoded_loc,
                "No. of Bedrooms": int(bedrooms),
                "Resale": is_resale,
            }
            input_dict.update(amenity_values)

            predicted_inr = predict_price(model, scaler, feature_cols, input_dict)

            # Format price into Crores or Lakhs
            if predicted_inr >= 1e7:
                cr_val = predicted_inr / 1e7
                price_badge = f"₹{cr_val:.2f} Crores"
            else:
                lakh_val = predicted_inr / 1e5
                price_badge = f"₹{lakh_val:.2f} Lakhs"

            st.markdown(
                f"""
                <div style="background: rgba(6, 78, 59, 0.25); border: 1px solid #10b981; border-radius: 12px; padding: 1.35rem; margin-top: 1.25rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <span style="font-size: 1.7rem;">🏷️</span>
                            <div>
                                <div style="font-size: 0.8rem; color: #a7f3d0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                                    Estimated Valuation
                                </div>
                                <div style="font-size: 1.75rem; font-weight: 800; color: #f8fafc; font-family: monospace;">
                                    ₹{predicted_inr:,.0f}
                                </div>
                            </div>
                        </div>
                        <span style="background: linear-gradient(90deg, #059669 0%, #10b981 100%); color: #ffffff; padding: 0.35rem 0.95rem; border-radius: 9999px; font-weight: 700; font-size: 0.95rem; font-family: monospace; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);">
                            {price_badge}
                        </span>
                    </div>
                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(16, 185, 129, 0.2); display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.82rem; color: #a7f3d0;">
                        <span>📍 <strong>Location:</strong> {selected_location}</span>
                        <span>📐 <strong>Area:</strong> {area:,} sq. ft.</span>
                        <span>🛏️ <strong>BHK:</strong> {bedrooms} Bedrooms</span>
                        <span>🏢 <strong>Type:</strong> {resale_option}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • Linear Regression & StandardScaler (Joblib)</span>"
        "<span>Data: Housing Prices in Metropolitan Areas of India (Kaggle)</span>"
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
                    (<code>housing_price_model.joblib</code> and <code>housing_scaler.joblib</code>) are loaded via 
                    <strong>joblib</strong> from the <code>models/</code> directory, bypassing runtime model training.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_data:</strong> Caches location label encodings and ordered feature column mappings.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">StandardScaler Normalization:</strong> Feature vectors are transformed with the exact scaling parameters learned during notebook training.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Unit Conversion:</strong> Target outputs in millions are rescaled back to Indian Rupees (INR) for intuitive presentation.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Relative Data Path:</strong> Resolves <code>projects/housing-price-prediction/data/Delhi.csv</code> relative to workspace root.
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
                    <code style="color: #10b981; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">Delhi.csv</code>
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
        "kaggle datasets download -d mlg-ulb/creditcardfraud -p projects/housing-price-prediction/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 <code>download_data.py</code> automatically skips re-downloading if <code>Delhi.csv</code> already exists in the <code>data/</code> folder.
        </p>
        """,
        unsafe_allow_html=True,
    )