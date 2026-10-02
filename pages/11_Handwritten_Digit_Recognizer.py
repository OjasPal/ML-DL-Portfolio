import os
import keras
from keras.layers import Dense
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------- Page Config ----------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Cyber Cyan Accent) ----------
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

    /* Metric Values & Labels - Cyber Cyan Glow */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(56, 189, 248, 0.4);
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
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Cyber Cyan Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #0284c7 100%) !important;
        box-shadow: 0 6px 22px rgba(56, 189, 248, 0.5) !important;
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

    /* Canvas Frame */
    .canvas-container {
        background-color: #0d1322;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Paths Definition ----------
PROJECT_DIR = "projects/handwritten-digit-recognizer"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "mnist_model.keras")
FALLBACK_MODEL_PATHS = [
    MODEL_PATH,
    "models/mnist_model.keras",
    "mnist_model.keras",
    os.path.join(PROJECT_DIR, "models", "mnist_model.h5"),
    "mnist_model.h5",
]


# ---------- Data Loading (Cached so it only runs once) ----------
@st.cache_data(show_spinner="Loading and preprocessing MNIST dataset...")
def load_data():
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

    xtrain = xtrain / 255.0
    xtest = xtest / 255.0

    xtrain = xtrain.reshape(-1, 784)
    xtest = xtest.reshape(-1, 784)

    return xtrain, ytrain, xtest, ytest


# ---------- Model Loading / Training (Cached for Instant Inference) ----------
@st.cache_resource(show_spinner="Loading pre-compiled neural network...")
def get_model():
    # 1. Check if pre-compiled Keras model artifact exists
    model_file = next((p for p in FALLBACK_MODEL_PATHS if os.path.exists(p)), None)
    if model_file:
        try:
            return keras.models.load_model(model_file), True
        except Exception:
            pass

    # 2. Otherwise train the Dense neural network architecture from the notebook
    xtrain, ytrain, xtest, ytest = load_data()

    model = keras.Sequential()
    model.add(Dense(64, activation="relu", input_dim=784))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(xtrain, ytrain, epochs=5, verbose=0)

    # Cache locally to speed up future dashboard runs
    try:
        os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
        model.save(MODEL_PATH)
    except Exception:
        pass

    return model, False


def preprocess_canvas_image(image_data):
    # Convert RGBA canvas array to grayscale, resize to 28x28 to match training data
    img = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 784), img


# ---------- Initialize Model ----------
model, loaded_from_file = get_model()

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
                <span style="font-size: 2rem;">✍️</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Handwritten Digit Recognizer
                </h1>
                <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #0284c7; color: #7dd3fc; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 11
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Draw any single digit (0–9) on the interactive dark canvas below. A Deep Neural Network (MLP) 
                trained on the <strong>MNIST</strong> dataset processes the normalized 28×28 pixel matrix and classifies the drawn stroke in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Performance & Architecture Metrics Card
    with st.expander("⚡ Neural Network Architecture & Dataset Metrics", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Fully connected Multi-Layer Perceptron: 784 Inputs → Dense(64, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)."
            "</p>",
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="INPUT RESOLUTION", value="28 × 28 (784 px)")
        with col_m2:
            st.metric(label="TARGET CLASSES", value="10 Digits (0–9)")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Drawing Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Draw a Digit (0–9)"
        "</h3>",
        unsafe_allow_html=True,
    )

    col_canvas, col_preview = st.columns([1.1, 0.9], gap="medium")

    with col_canvas:
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.4rem;'>"
            "Draw a centered digit using white strokes on the black canvas:"
            "</p>",
            unsafe_allow_html=True,
        )
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=18,
            stroke_color="white",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="mnist_canvas",
            return_image_data=True,
        )

    with col_preview:
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.4rem;'>"
            "Preprocessed Model Input (Resized to 28×28 Grayscale):"
            "</p>",
            unsafe_allow_html=True,
        )
        if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
            _, preview_img = preprocess_canvas_image(canvas_result.image_data)
            st.image(preview_img, width=140, caption="28×28 Normalized Matrix")
        else:
            st.markdown(
                """
                <div style="width: 140px; height: 140px; background: #0d1322; border: 1px dashed #1e293b; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 0.75rem; text-align: center; padding: 0.5rem;">
                    Awaiting canvas drawing...
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    predict_clicked = st.button("⚡ Predict Digit")

    if predict_clicked:
        if canvas_result.image_data is None or canvas_result.image_data[:, :, :3].sum() == 0:
            st.warning("Please draw a digit on the canvas before predicting.")
        else:
            input_data, _ = preprocess_canvas_image(canvas_result.image_data)
            prediction = model.predict(input_data, verbose=0)
            predicted_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            # Styled Prediction Output Banner
            st.markdown(
                f"""
                <div style="background: rgba(2, 132, 199, 0.2); border: 1px solid #38bdf8; border-radius: 12px; padding: 1.35rem; margin-top: 1.25rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 0.85rem;">
                            <span style="font-size: 2.2rem; background: rgba(56, 189, 248, 0.2); width: 56px; height: 56px; display: flex; align-items: center; justify-content: center; border-radius: 10px; font-weight: 800; color: #38bdf8; font-family: monospace;">
                                {predicted_digit}
                            </span>
                            <div>
                                <div style="font-size: 0.8rem; color: #7dd3fc; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                                    Neural Network Classification
                                </div>
                                <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc;">
                                    Predicted Digit: {predicted_digit}
                                </div>
                            </div>
                        </div>
                        <span style="background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%); color: #ffffff; padding: 0.35rem 0.95rem; border-radius: 9999px; font-weight: 700; font-size: 0.9rem; font-family: monospace; box-shadow: 0 4px 14px rgba(56, 189, 248, 0.35);">
                            {confidence:.1f}% Confidence
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Class Probability Distribution
            st.markdown(
                "<h4 style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-top: 1.25rem; margin-bottom: 0.5rem;'>"
                "Softmax Probability Distribution Across All Digits (0–9):"
                "</h4>",
                unsafe_allow_html=True,
            )

            prob_df = pd.DataFrame(
                {
                    "Digit": [str(d) for d in range(10)],
                    "Probability": prediction[0],
                }
            ).set_index("Digit")

            st.bar_chart(prob_df, color="#38bdf8")

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • Keras Neural Network (MLP)</span>"
        "<span>Data: MNIST Handwritten Digits Dataset</span>"
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
                    <strong style="color: #e2e8f0;">@st.cache_resource:</strong> Caches the compiled <strong>Keras</strong> Sequential model 
                    (<code>Dense(64) &rarr; Dense(64) &rarr; Dense(10)</code>) so neural inference runs instantaneously.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_data:</strong> Caches the MNIST 60k training samples and normalization (<code>/ 255.0</code>).
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Drawable Canvas:</strong> Uses <code>streamlit-drawable-canvas</code> at 280&times;280 resolution, downsampling to 28&times;28 before inference.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Softmax Distribution:</strong> Displays the complete 10-class probability spread via <code>st.bar_chart</code>.
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
                    <code style="color: #38bdf8; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">mnist.npz / keras.datasets.mnist</code>
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
        "kaggle datasets download -d oddrationale/mnist-in-csv -p projects/handwritten-digit-recognizer/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 The script can also automatically stream from <code>keras.datasets.mnist</code> with zero manual downloading required.
        </p>
        """,
        unsafe_allow_html=True,
    )