import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------- Page config ----------
st.set_page_config(page_title="Delhi Housing Price Predictor", page_icon="🏠", layout="centered")

DATA_PATH = "projects/housing-price-prediction/data/Delhi.csv"


# ---------- Data loading + model building (cached so it only runs once) ----------
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    # Resolve relative path for root execution or fallback to local data folder
    path = DATA_PATH
    if not os.path.exists(path):
        if os.path.exists("data/Delhi.csv"):
            path = "data/Delhi.csv"
        elif os.path.exists("Delhi.csv"):
            path = "Delhi.csv"

    housing = pd.read_csv(path)

    # Encode Location as in original notebook
    label_encode = LabelEncoder()
    housing["Location_Encoded"] = label_encode.fit_transform(housing["Location"])

    # Target variable (Price) scaled to millions as in notebook
    housing["Price_in_Millions"] = housing["Price"] / 1e6

    locations_list = sorted(housing["Location"].unique().tolist())
    return housing, label_encode, locations_list


@st.cache_resource(show_spinner="Training model... (this happens once)")
def train_model(housing):
    # Prepare features matching notebook's exact columns & order
    feature_df = housing.drop(["Price", "Price_in_Millions", "Location_Encoded"], axis=1, errors="ignore").copy()
    feature_df["Location"] = housing["Location_Encoded"]

    # Stratified split on Resale
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(feature_df, feature_df["Resale"]):
        strat_train_features = feature_df.loc[train_index]
        strat_train_target = housing.loc[train_index, "Price_in_Millions"]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(strat_train_features)
    y_train = strat_train_target

    model = LinearRegression()
    model.fit(x_train, y_train)

    feature_cols = list(strat_train_features.columns)
    return model, scaler, feature_cols


def predict_price(model, scaler, feature_cols, input_values):
    # Build input row in exact order of model features
    row = np.array([[input_values[col] for col in feature_cols]])
    scaled_row = scaler.transform(row)
    price_millions = model.predict(scaled_row)[0]
    # Price is in millions; multiply by 1e6 to get INR
    price_inr = price_millions * 1e6
    return max(0.0, price_inr)


# ---------- UI ----------
st.title("🏠 Delhi Housing Price Prediction")
st.write(
    "Estimate apartment and house prices across Delhi based on area, location, bedrooms, and amenities "
    "using a Linear Regression model trained on historical real estate data."
)

if not os.path.exists(DATA_PATH) and not os.path.exists("data/Delhi.csv") and not os.path.exists("Delhi.csv"):
    st.error(
        f"Dataset not found at `{DATA_PATH}`. Please run `download_data.py` inside the project folder first."
    )
    st.stop()

housing, label_encoder, locations_list = load_data()
model, scaler, feature_cols = train_model(housing)

st.subheader("Property Details")
col1, col2 = st.columns(2)

with col1:
    selected_location = st.selectbox("Location", options=locations_list, index=0)
    area = st.number_input("Area (in sq. ft.)", min_value=100, max_value=15000, value=1200, step=50)

with col2:
    bedrooms = st.selectbox("Number of Bedrooms", options=[1, 2, 3, 4, 5, 6, 7, 8], index=1)
    resale_option = st.radio("Property Condition", options=["Resale Property", "New / Fresh Construction"], index=0)
    is_resale = 1 if resale_option == "Resale Property" else 0

st.subheader("Amenities & Features")
with st.expander("Select Available Amenities", expanded=False):
    amenity_cols = [
        col for col in feature_cols
        if col not in ["Area", "Location", "No. of Bedrooms", "Resale"]
    ]

    amenity_values = {}
    sub_col1, sub_col2, sub_col3 = st.columns(3)

    for i, amenity in enumerate(amenity_cols):
        target_col = [sub_col1, sub_col2, sub_col3][i % 3]
        clean_name = amenity.replace("24X7", "24x7 ").replace("BED", "Bed Available")
        with target_col:
            amenity_values[amenity] = 1 if st.checkbox(clean_name, value=False) else 0

if st.button("Predict Price", type="primary"):
    if area <= 0:
        st.warning("Please enter a valid property area in square feet.")
    else:
        encoded_location = label_encoder.transform([selected_location])[0]

        input_dict = {
            "Area": float(area),
            "Location": encoded_location,
            "No. of Bedrooms": int(bedrooms),
            "Resale": is_resale,
        }
        input_dict.update(amenity_values)

        predicted_price = predict_price(model, scaler, feature_cols, input_dict)

        st.success(f"### Predicted Price: ₹{predicted_price:,.0f}")

        if predicted_price >= 1e7:
            st.info(f"Approximately **₹{predicted_price / 1e7:.2f} Crores**")
        else:
            st.info(f"Approximately **₹{predicted_price / 1e5:.2f} Lakhs**")

st.divider()
st.caption("Built with Streamlit • Data: Housing Prices in Metropolitan Areas of India (Kaggle)")