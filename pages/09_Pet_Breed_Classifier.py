import io
import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# ---------- Page Config ----------
st.set_page_config(
    page_title="Pet Breed Classifier",
    page_icon="🐾",
    layout="wide",
)


# ---------- Custom CSS (Unified Portfolio Theme - Deep Teal Accent) ----------
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

    /* Metric Values & Labels - Deep Teal Glow */
    [data-testid="stMetricValue"] {
        color: #2dd4bf !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(45, 212, 191, 0.4);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Input & Select Elements */
    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #0d1322 !important;
        border: 1px dashed #1e293b !important;
        border-radius: 10px !important;
    }

    /* Deep Teal Gradient "Browse files" Button with Hover Glow */
    section[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #0f766e 0%, #14b8a6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(15, 118, 110, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }

    section[data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(90deg, #115e59 0%, #0f766e 100%) !important;
        box-shadow: 0 6px 22px rgba(45, 212, 191, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Slider accent */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #14b8a6 !important;
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

    .ais-card code {
        color: #5eead4;
        background: #0d1322;
        padding: 0.12rem 0.3rem;
        border-radius: 4px;
    }

    .ais-card a {
        color: #2dd4bf;
        text-decoration: none;
        word-break: break-all;
    }

    /* Dark Code Box Below Resource Card */
    .code-box-container {
        background-color: #0d1322;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-family: monospace;
        font-size: 0.78rem;
        color: #2dd4bf;
        margin-top: -0.5rem;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }

    .code-box-comment {
        color: #64748b;
    }

    /* Project metadata rows */
    .meta-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.45rem 0;
        border-bottom: 1px solid #1e293b;
        font-size: 0.82rem;
    }

    .meta-row:last-child {
        border-bottom: none;
    }

    .meta-row span:first-child {
        color: #94a3b8;
    }

    .meta-row span:last-child {
        color: #e2e8f0;
        font-weight: 600;
        text-align: right;
    }

    /* Species badge (CAT / DOG) */
    .species-badge {
        display: inline-block;
        background: rgba(45, 212, 191, 0.12);
        border: 1px solid #14b8a6;
        color: #5eead4;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        box-shadow: 0 0 18px rgba(45, 212, 191, 0.25);
    }

    /* Predicted breed card */
    .breed-card {
        background: rgba(20, 184, 166, 0.10);
        border: 1px solid #14b8a6;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.9rem 0;
    }

    .breed-label {
        font-size: 0.78rem;
        color: #5eead4;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .breed-name {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0.2rem 0 0.7rem 0;
    }

    .conf-pill {
        background: linear-gradient(90deg, #0f766e 0%, #14b8a6 100%);
        color: #ffffff;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
        font-family: monospace;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.35);
    }

    /* Selectbox / uploader / radio text */
    .stSelectbox label,
    .stSlider label,
    .stRadio label,
    .stFileUploader label {
        color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Project / Model Paths ----------
PROJECT_DIR = "projects/pet-breed-classifier"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "pet_breed_classifier.pth")

SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")

SAMPLE_IMAGES = {
    "Cat Sample 1": os.path.join(SAMPLES_DIR, "abyssinian_cat.jpg"),
    "Cat Sample 2": os.path.join(SAMPLES_DIR, "ragdoll_cat.jpg"),
    "Dog Sample 1": os.path.join(SAMPLES_DIR, "siberian_husky.jpg"),
    "Dog Sample 2": os.path.join(SAMPLES_DIR, "chihuahua.jpg"),
}

GITHUB_URL = "https://github.com/OjasPal/ML-DL-Portfolio"

# The 12 cat breeds (every other class in the model is a dog breed)
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
}

# Folder name used during training -> nicer name shown in the app
DISPLAY_NAMES = {"Pembroke": "Pembroke Welsh Corgi"}

TOP_K = 5


# ---------- Model Loading ----------
@st.cache_resource(show_spinner=False)
def get_classifier():
    """
    Loads the saved EfficientNet-B0 artifact once (weights, class names and
    preprocessing stats), so the model is never retrained or rebuilt on a rerun.
    """
    artifact = torch.load(MODEL_PATH, map_location="cpu")
    class_names = artifact["class_names"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(artifact["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(artifact["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(artifact["mean"], artifact["std"]),
    ])

    return {
        "model": model,
        "class_names": class_names,
        "transform": transform,
        "device": device,
        "backend_name": "CUDA / GPU" if device.type == "cuda" else "CPU",
        "test_accuracy": artifact.get("test_accuracy"),
    }


model_error = None
classifier = None

if not os.path.exists(MODEL_PATH):
    model_error = (
        f"Model file not found at `{MODEL_PATH}`. "
        "Run the last cell of the notebook (Saving the Model) to create it."
    )
else:
    try:
        classifier = get_classifier()
    except Exception as exc:
        model_error = f"The model file at `{MODEL_PATH}` could not be loaded: {exc}"


# ---------- Image Helpers ----------
@st.cache_data(show_spinner=False)
def load_sample_bytes(path):
    with open(path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def decode_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)      # fixes sideways phone photos
    return image.convert("RGB")


def display_name(breed):
    return DISPLAY_NAMES.get(breed, breed)


# ---------- Classification Logic ----------
@st.cache_data(show_spinner=False)
def classify_image(image_bytes):
    """
    Runs the fine-tuned EfficientNet-B0 on an image buffer and returns the
    Top-5 (breed, probability) pairs plus the inference time in milliseconds.
    """
    net = get_classifier()
    image = decode_image(image_bytes)

    tensor = net["transform"](image).unsqueeze(0).to(net["device"])

    start_time = time.perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(net["model"](tensor), dim=1)[0].cpu()
    inference_ms = (time.perf_counter() - start_time) * 1000

    top_probs, top_idx = probabilities.topk(TOP_K)
    top_predictions = [
        (net["class_names"][int(i)], float(p)) for p, i in zip(top_probs, top_idx)
    ]

    return top_predictions, inference_ms


def build_top5_chart(top_predictions):
    """Horizontal bar chart of the Top-5 breeds, best prediction highlighted in teal."""
    chart_df = pd.DataFrame(
        {
            "Breed": [display_name(name) for name, _ in top_predictions],
            "Probability": [prob for _, prob in top_predictions],
        }
    )

    fig = px.bar(
        chart_df,
        x="Probability",
        y="Breed",
        orientation="h",
        text=chart_df["Probability"].map(lambda p: f"{p:.1%}"),
    )
    fig.update_traces(
        marker_color=["#2dd4bf"] + ["#0f766e"] * (len(chart_df) - 1),
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_xaxes(visible=False, range=[0, 1.25])
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        showlegend=False,
    )
    return fig


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
                <span style="font-size: 2rem;">🐾</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Pet Breed Classifier
                </h1>
                <span style="background: rgba(45, 212, 191, 0.12); border: 1px solid #0f766e; color: #5eead4; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 9
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Identify <strong>46 cat and dog breeds</strong> instantly using a fine-tuned
                <strong>EfficientNet-B0</strong> deep learning model.
                Upload a pet photo or pick a sample image to see the predicted breed and Top-5 confidence scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(model_error)
        st.stop()

    # 1. In-page Classification Controls Block
    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 1rem 1.15rem; margin-bottom: 1rem;">
            <div style="font-size: 0.98rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.2rem;">
                🎛️ Classification Controls
            </div>
            <div style="color: #64748b; font-size: 0.76rem;">
                Choose an input source and how confident the model must be before a prediction is trusted.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col_1, control_col_2 = st.columns(2, gap="medium")

    with control_col_1:
        input_mode = st.radio(
            "Input Source",
            ["Upload Image", "Try Sample Images"],
            index=1,
            horizontal=True,
        )

    with control_col_2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help="Predictions below this confidence are flagged as uncertain.",
        )

    selected_sample = None
    uploaded_file = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a Pet Photo",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG.",
        )
    else:
        selected_sample = st.selectbox(
            "Choose a Sample Image",
            list(SAMPLE_IMAGES.keys()),
        )

    # 2. Get Current Input
    input_bytes = None
    input_label = None
    input_load_error = None

    try:
        if input_mode == "Try Sample Images" and selected_sample:
            input_bytes = load_sample_bytes(SAMPLE_IMAGES[selected_sample])
            input_label = selected_sample

        elif input_mode == "Upload Image" and uploaded_file is not None:
            input_bytes = uploaded_file.getvalue()
            input_label = uploaded_file.name

    except Exception as exc:
        input_load_error = str(exc)

    if input_load_error:
        st.error(f"Could not load the selected image: {input_load_error}")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 3. Model Performance Metrics Expander
    with st.expander("⚡ Model Performance Metrics", expanded=True):
        st.markdown(
            """
            <p style="color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;">
                EfficientNet-B0 fine-tuned in 2 stages: classifier head only, then the full network.
            </p>
            """,
            unsafe_allow_html=True,
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        with metric_1:
            st.metric(label="MODEL", value="EfficientNet-B0")

        with metric_2:
            st.metric(label="VALIDATION ACCURACY", value="93.6%")

        with metric_3:
            test_accuracy = classifier["test_accuracy"]
            st.metric(
                label="TEST ACCURACY",
                value=f"{test_accuracy * 100:.1f}%" if test_accuracy is not None else "N/A",
            )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 4. Input Preview / Prediction Section
    st.markdown(
        """
        <h3 style="font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;">
            Identify the Breed
        </h3>
        """,
        unsafe_allow_html=True,
    )

    if input_bytes is None:
        st.markdown(
            """
            <div style="background: #111827; border: 1px dashed #1e293b; border-radius: 10px; padding: 2rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.6rem;">🐶🐱</div>
                <div style="font-size: 1rem; color: #cbd5e1; font-weight: 600;">
                    No image selected yet
                </div>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;">
                    Upload a JPG or PNG photo of a cat or dog, or choose a sample image above.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        try:
            pet_image = decode_image(input_bytes)
            image_width, image_height = pet_image.size

            top_predictions, inference_ms = classify_image(input_bytes)
            top_breed, top_prob = top_predictions[0]

            preview_col, result_col = st.columns(2, gap="large")

            # ----- Left: image preview -----
            with preview_col:
                st.markdown(
                    f"""
                    <p style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.5rem;">
                        Current Input: <strong style="color: #cbd5e1;">{input_label}</strong>
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(
                    pet_image,
                    caption=f"{image_width} × {image_height} px",
                    use_container_width=True,
                )

            # ----- Right: prediction results -----
            with result_col:
                species_icon, species_label = ("🐱", "CAT") if top_breed in CAT_BREEDS else ("🐶", "DOG")

                # Species Badge
                st.markdown(
                    f'<div class="species-badge">{species_icon} {species_label}</div>',
                    unsafe_allow_html=True,
                )

                # Predicted Breed Card
                st.markdown(
                    f"""
                    <div class="breed-card">
                        <div class="breed-label">Predicted Breed</div>
                        <div class="breed-name">{display_name(top_breed)}</div>
                        <span class="conf-pill">{top_prob * 100:.1f}% Confidence</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Top-5 Predictions Chart (Clean & Static with displayModeBar=False)
                st.markdown(
                    "<p style='color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.2rem;'>Top-5 Predictions</p>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    build_top5_chart(top_predictions),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                # Confidence Alert
                if top_prob < confidence_threshold:
                    st.warning(
                        f"Low confidence: {top_prob * 100:.1f}% is below your "
                        f"{confidence_threshold * 100:.0f}% threshold. "
                        "Try a clearer, well-lit photo where the pet's face is visible."
                    )

                result_m1, result_m2 = st.columns(2)

                with result_m1:
                    st.metric(label="INFERENCE TIME", value=f"{inference_ms:.1f} ms")

                with result_m2:
                    st.metric(label="BACKEND", value=classifier["backend_name"])

        except Exception as exc:
            st.error(f"Unable to process this image: {exc}")

    # Footer
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <span>Built with Streamlit • PyTorch EfficientNet-B0</span>
            <span>Data: Oxford-IIIT Pet + Stanford Dogs</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# RIGHT-HAND INFO COLUMN (SIDEBAR / DOCUMENTATION CARD)
# =========================================================================
with col_info:
    # Project Metadata Card
    st.markdown(
        """
        <div class="ais-card">
            <h4>
                <span>🐾</span> Project Metadata
            </h4>
            <div class="meta-row"><span>Model Backbone</span><span>EfficientNet-B0</span></div>
            <div class="meta-row"><span>Validation Accuracy</span><span>93.6%</span></div>
            <div class="meta-row"><span>Dataset Size</span><span>8,847 Images</span></div>
            <div class="meta-row"><span>Total Classes</span><span>46 Breeds: 12 Cats / 34 Dogs</span></div>
            <div class="meta-row"><span>Framework</span><span>PyTorch / CUDA</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Notebook to Streamlit Conversion Card
    st.markdown(
        """
        <div class="ais-card">
            <h4>
                <span>⚙️</span> Notebook to Streamlit Conversion
            </h4>
            <ul style="color: #94a3b8; font-size: 0.82rem; line-height: 1.6; padding-left: 1.15rem; margin-bottom: 0;">
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_resource:</strong>
                    Loads the saved <code>pet_breed_classifier.pth</code> artifact (EfficientNet-B0 weights,
                    class names and normalization stats) once, so the model is never retrained.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Preprocessing:</strong>
                    Same as the notebook: <code>Resize(256)</code>, <code>CenterCrop(224)</code> and ImageNet normalization.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Prediction:</strong>
                    Softmax over 46 breeds. The Top-5 are charted and the threshold slider flags uncertain predictions.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Device:</strong>
                    Uses the GPU (CUDA) when available and falls back to the CPU.
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset & Resources Card
    github_artifact_link = ""
    if GITHUB_URL:
        github_artifact_link = f'<a href="{GITHUB_URL}/blob/main/{PROJECT_DIR}/models/pet_breed_classifier.pth" target="_blank">pet_breed_classifier.pth</a>'
    else:
        github_artifact_link = '<code>models/pet_breed_classifier.pth</code>'

    st.markdown(
        f"""
        <div class="ais-card" style="margin-bottom: 0.75rem;">
            <h4>
                <span>📥</span> Dataset & Resources
            </h4>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                <div style="margin-bottom: 0.65rem;">
                    <span style="color: #cbd5e1; font-weight: 500;">Dataset Sources:</span><br/>
                    • <strong>Oxford-IIIT Pet:</strong><br/>
                    <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/" target="_blank">https://www.robots.ox.ac.uk/~vgg/data/pets/</a><br/>
                    • <strong>Stanford Dogs:</strong><br/>
                    <a href="http://vision.stanford.edu/aditya86/ImageNetDogs/main.html" target="_blank">http://vision.stanford.edu/aditya86/ImageNetDogs/main.html</a>
                </div>
                <div>
                    <span style="color: #cbd5e1; font-weight: 500;">Artifact Target:</span><br/>
                    {github_artifact_link}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dark Gray Container Code Box (Matches Face Detection layout)
    st.markdown(
        """
        <div class="code-box-container">
            <span class="code-box-comment"># Loaded automatically on app launch</span><br/>
            torch.load("models/pet_breed_classifier.pth")
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Informative note with 💡 emoji
    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: 0.25rem; line-height: 1.4;">
            💡 The fine-tuned EfficientNet-B0 weights and class dictionary are loaded once into memory using <code>@st.cache_resource</code> for rapid inference.
        </p>
        """,
        unsafe_allow_html=True,
    )