import io
import os
import time
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Face Detector",
    page_icon="🧑‍💻",
    layout="wide",
)


# ---------- Custom CSS (Unified Portfolio Theme - Violet Indigo Accent) ----------
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

    /* Metric Values & Labels - Violet Indigo Glow */
    [data-testid="stMetricValue"] {
        color: #818cf8 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(129, 140, 248, 0.4);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Input & Select Elements */
    .stTextInput input,
    .stTextArea textarea {
        background-color: #0d1322 !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }

    /* Violet Indigo Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(67, 56, 202, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #3730a3 0%, #4f46e5 100%) !important;
        box-shadow: 0 6px 22px rgba(129, 140, 248, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #0d1322 !important;
        border: 1px dashed #1e293b !important;
        border-radius: 10px !important;
    }

    /* Slider accent */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #6366f1 !important;
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
        color: #a5b4fc;
        background: #0d1322;
        padding: 0.12rem 0.3rem;
        border-radius: 4px;
    }

    /* Selectbox / uploader text */
    .stSelectbox label,
    .stSlider label,
    .stFileUploader label,
    .stCameraInput label {
        color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Project / Model Paths ----------
PROJECT_DIR = "projects/face-detection"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

PROTO_FILE = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"

PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")

SAMPLE_IMAGES = {
    "Single Portrait": os.path.join(SAMPLES_DIR, "single_portrait.jpg"),
    "Group Photo (5 People)": os.path.join(SAMPLES_DIR, "group_photo.jpg"),
    "Team / Office Setting": os.path.join(SAMPLES_DIR, "team_office.jpg"),
}


# ---------- Model Download / Loading ----------
def _download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    urllib.request.urlretrieve(url, destination)


@st.cache_resource(show_spinner="Loading pre-trained face detection model...")
def get_face_detector():
    """
    Downloads the OpenCV Caffe SSD model artifacts when missing,
    then loads the ResNet-10 SSD detector.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    proto_path = os.path.join(MODEL_DIR, PROTO_FILE)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)

    if not os.path.exists(proto_path):
        _download_file(PROTO_URL, proto_path)

    if not os.path.exists(model_path):
        _download_file(MODEL_URL, model_path)

    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    backend_name = "CPU"
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            backend_name = "CUDA / GPU"
    except Exception:
        backend_name = "CPU"

    return net, backend_name


@st.cache_data(show_spinner=False)
def load_sample_bytes(path):
    with open(path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def decode_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


# ---------- Face Detection Logic ----------
@st.cache_data(show_spinner=False)
def detect_faces(image_bytes, confidence_threshold):
    """
    Runs the ResNet-10 SSD Caffe face detector on an image buffer.
    Uses 300x300 blob, mean subtraction (104, 177, 123), and draws green bounding boxes.
    """
    net, backend_name = get_face_detector()

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("The selected image could not be decoded.")

    height, width = img_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )

    start_time = time.perf_counter()

    net.setInput(blob)
    detections = net.forward()

    inference_ms = (time.perf_counter() - start_time) * 1000

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = []

    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array(
                [width, height, width, height]
            )

            start_x, start_y, end_x, end_y = box.astype("int")

            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(width - 1, end_x), min(height - 1, end_y)

            faces.append(
                {
                    "box": (
                        start_x,
                        start_y,
                        end_x - start_x,
                        end_y - start_y,
                    ),
                    "confidence": confidence,
                }
            )

            cv2.rectangle(
                img_rgb,
                (start_x, start_y),
                (end_x, end_y),
                (0, 255, 0),
                3,
            )

    return (
        img_rgb,
        faces,
        inference_ms,
        backend_name,
    )


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
                <span style="font-size: 2rem;">🧑‍💻</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    AI Face Detector
                </h1>
                <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid #4f46e5; color: #a5b4fc; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 10
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Multi-face detection in images and webcam snapshots using
                <strong>OpenCV Deep Neural Networks (ResNet-10 SSD Caffe Model)</strong>.
                Select a sample, upload an image, or capture a webcam snapshot to detect human faces.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. In-page Detection Controls Block
    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 1rem 1.15rem; margin-bottom: 1rem;">
            <div style="font-size: 0.98rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.2rem;">
                🎛️ Detection Controls
            </div>
            <div style="color: #64748b; font-size: 0.76rem;">
                Choose an input source and detection sensitivity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col_1, control_col_2 = st.columns(2, gap="medium")

    with control_col_1:
        input_mode = st.selectbox(
            "Input Mode",
            ["Sample Image", "Upload Image", "Live Camera"],
        )

    with control_col_2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only detections above this confidence are shown.",
        )

    selected_sample = None
    uploaded_file = None
    camera_file = None

    if input_mode == "Sample Image":
        selected_sample = st.selectbox(
            "Choose a Sample Image",
            list(SAMPLE_IMAGES.keys()),
        )

    elif input_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG.",
        )

    else:
        camera_file = st.camera_input(
            "Capture a Snapshot",
        )

    # 2. Get Current Input
    input_bytes = None
    input_label = None
    input_load_error = None

    try:
        if input_mode == "Sample Image" and selected_sample:
            input_bytes = load_sample_bytes(SAMPLE_IMAGES[selected_sample])
            input_label = selected_sample

        elif input_mode == "Upload Image" and uploaded_file is not None:
            input_bytes = uploaded_file.getvalue()
            input_label = uploaded_file.name

        elif input_mode == "Live Camera" and camera_file is not None:
            input_bytes = camera_file.getvalue()
            input_label = "Live Camera Snapshot"

    except Exception as exc:
        input_load_error = str(exc)

    if input_load_error:
        st.error(f"Could not load the selected image: {input_load_error}")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 3. Model / Runtime Metrics Expander
    with st.expander("⚡ Face Detection Model & Runtime Metrics", expanded=True):
        st.markdown(
            """
            <p style="color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;">
                Pre-trained ResNet-10 SSD face detector using OpenCV DNN with 300×300 blob preprocessing.
            </p>
            """,
            unsafe_allow_html=True,
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        with metric_1:
            st.metric(label="MODEL", value="ResNet-10 SSD")

        with metric_2:
            st.metric(label="INPUT SIZE", value="300 × 300")

        with metric_3:
            st.metric(
                label="THRESHOLD",
                value=f"{confidence_threshold:.2f}",
            )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 4. Input / Detection Section
    st.markdown(
        """
        <h3 style="font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;">
            Detect Faces in Image
        </h3>
        """,
        unsafe_allow_html=True,
    )

    if input_bytes is None:
        st.markdown(
            """
            <div style="background: #111827; border: 1px dashed #1e293b; border-radius: 10px; padding: 2rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.6rem;">🖼️</div>
                <div style="font-size: 1rem; color: #cbd5e1; font-weight: 600;">
                    No image selected yet
                </div>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;">
                    Choose a sample, upload an image, or capture a camera snapshot above.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        try:
            original_image = decode_image(input_bytes)

            st.markdown(
                f"""
                <p style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.5rem;">
                    Current Input: <strong style="color: #cbd5e1;">{input_label}</strong>
                </p>
                """,
                unsafe_allow_html=True,
            )

            preview_col, action_col = st.columns([3.2, 1.0], gap="medium")

            with preview_col:
                st.image(
                    original_image,
                    caption="Original Input Image",
                    use_container_width=True,
                )

            with action_col:
                st.markdown(
                    """
                    <div style="height: 0.3rem;"></div>
                    <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.5; margin-bottom: 0.8rem;">
                        Run the pre-trained SSD detector on this image buffer.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                detect_clicked = st.button(
                    "⚡ Detect Faces",
                    use_container_width=True,
                )

            if detect_clicked:
                processed_image, faces, inference_ms, backend_name = detect_faces(
                    input_bytes,
                    confidence_threshold,
                )

                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

                # Detection Result Banner
                if faces:
                    banner_background = "rgba(99, 102, 241, 0.18)"
                    banner_border = "#818cf8"
                    status_icon = "✅"
                    status_text = f"{len(faces)} Face{'s' if len(faces) != 1 else ''} Detected"
                    status_subtext = "Human faces were detected above the selected confidence threshold."
                else:
                    banner_background = "rgba(245, 158, 11, 0.10)"
                    banner_border = "#f59e0b"
                    status_icon = "⚠️"
                    status_text = "No Faces Detected"
                    status_subtext = "Try lowering the confidence threshold or use an image with clearer faces."

                st.markdown(
                    f"""
                    <div style="background: {banner_background}; border: 1px solid {banner_border}; border-radius: 12px; padding: 1.2rem; margin-top: 0.2rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                            <div style="display: flex; align-items: center; gap: 0.8rem;">
                                <span style="font-size: 1.7rem;">{status_icon}</span>
                                <div>
                                    <div style="font-size: 0.78rem; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                                        Detection Result
                                    </div>
                                    <div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc;">
                                        {status_text}
                                    </div>
                                </div>
                            </div>
                            <span style="background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%); color: #ffffff; padding: 0.35rem 0.85rem; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; font-family: monospace; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);">
                                {inference_ms:.1f} ms
                            </span>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 0.65rem; line-height: 1.4;">
                            {status_subtext}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)

                # Comparison View
                comparison_left, comparison_right = st.columns(2, gap="medium")

                with comparison_left:
                    st.markdown(
                        "<p style='color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.4rem;'>Original Input Image</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(
                        original_image,
                        caption=input_label,
                        use_container_width=True,
                    )

                with comparison_right:
                    st.markdown(
                        "<p style='color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.4rem;'>Processed Output — Green Bounding Boxes</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(
                        processed_image,
                        caption=f"{len(faces)} face(s) detected",
                        use_container_width=True,
                    )

                st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

                result_m1, result_m2, result_m3 = st.columns(3)

                with result_m1:
                    st.metric(
                        label="TOTAL FACES DETECTED",
                        value=str(len(faces)),
                    )

                with result_m2:
                    st.metric(
                        label="INFERENCE TIME",
                        value=f"{inference_ms:.1f} ms",
                    )

                with result_m3:
                    st.metric(
                        label="BACKEND",
                        value=backend_name,
                    )

                if not faces:
                    st.info(
                        "No faces crossed the current confidence threshold. "
                        "Try a lower threshold such as 0.35–0.45."
                    )

        except Exception as exc:
            st.error(f"Unable to process this image: {exc}")

    # Footer
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <span>Built with Streamlit • OpenCV DNN (ResNet-10 SSD)</span>
            <span>Model: Caffe Face Detector • 300×300 Input</span>
        </div>
        """,
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
                    <strong style="color: #e2e8f0;">@st.cache_resource:</strong>
                    Caches the loaded <strong>OpenCV DNN ResNet-10 SSD</strong> network so it is not rebuilt on every rerun.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Auto Model Download:</strong>
                    Downloads <code>deploy.prototxt</code> and
                    <code>res10_300x300_ssd_iter_140000.caffemodel</code> only when they are missing.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Preprocessing:</strong>
                    Uses the notebook's 300×300 SSD blob and mean subtraction
                    <code>(104, 177, 123)</code>.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Detection:</strong>
                    Filters confidence scores, clips boxes to image boundaries,
                    and draws green rectangles around detected faces.
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Artifacts & Weights Info Card
    st.markdown(
        """
        <div class="ais-card">
            <h4>
                <span>📥</span> Model Artifacts & Weights
            </h4>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                <div style="margin-bottom: 0.65rem;">
                    <span style="color: #cbd5e1; font-weight: 500;">Model Source:</span><br/>
                    <code style="color: #818cf8; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">OpenCV GitHub / opencv_3rdparty</code>
                </div>
                <div style="margin-bottom: 0.65rem;">
                    <span style="color: #cbd5e1; font-weight: 500;">Artifact Target:</span><br/>
                    <code style="color: #818cf8; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">deploy.prototxt / caffemodel</code>
                </div>
                <div>
                    <span style="color: #cbd5e1; font-weight: 500;">Python Download Script:</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        """# Automatically downloaded on first load if missing
urllib.request.urlretrieve(PROTO_URL, "models/deploy.prototxt")
urllib.request.urlretrieve(MODEL_URL, "models/res10_300x300_ssd_iter_140000.caffemodel")""",
        language="python",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 The pre-trained OpenCV Caffe model artifacts are automatically fetched and cached on first load with zero manual downloading required.
        </p>
        """,
        unsafe_allow_html=True,
    )