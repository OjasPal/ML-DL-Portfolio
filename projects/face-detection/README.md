# 🧑‍💻 AI Face Detector

An interactive face detection dashboard built with Streamlit, locating human faces in images using OpenCV's deep neural network module and a pre-trained ResNet-10 SSD model.

## Overview

Choose a built-in sample image, upload your own, or capture a webcam snapshot, then set a confidence threshold and run detection. The dashboard draws bounding boxes around every detected face and shows the original and annotated images side by side, along with the face count, inference time, and compute backend. The interface is built as a custom two-panel dashboard: a detection panel on the left, and a live architecture/documentation panel on the right.

This is a face **detection** tool: it finds where faces are in an image, and does not identify who they belong to. Images are processed in memory and are not written to disk by the app.

## Model

This project uses a pre-trained model rather than a dataset you need to download:

- **Architecture:** ResNet-10 backbone with a Single Shot MultiBox Detector (SSD) head, distributed with OpenCV's DNN samples in Caffe format
- **Model files:** `deploy.prototxt` (network definition) and `res10_300x300_ssd_iter_140000.caffemodel` (weights), sourced from the OpenCV GitHub repositories
- **Auto-download:** on first load, the app downloads any missing model file into `projects/face-detection/models/`, so no manual setup is needed

## How It Works

1. The input image is decoded and resized to a 300×300 blob, with mean subtraction `(104, 177, 123)` applied to match the model's training preprocessing.
2. The network runs a single forward pass and returns candidate detections, each with a confidence score and a normalized bounding box.
3. Detections above the selected confidence threshold are kept, scaled back to the original image dimensions, and clipped to the image boundaries.
4. Green bounding boxes are drawn on a copy of the image, and the result is shown next to the original with detection metrics.

The network is loaded once and cached with `@st.cache_resource`, so it isn't rebuilt on every interaction. Image decoding, sample fetching, and detection results are cached with `@st.cache_data`. If OpenCV is built with CUDA support and a compatible GPU is available, inference automatically runs on the GPU; otherwise it runs on CPU.

## Features

- Three input modes: sample images, image upload (JPG, JPEG, PNG), and webcam snapshot
- Adjustable confidence threshold (0.10–1.00) to trade off sensitivity against false positives
- Side-by-side original and annotated image comparison
- Live metrics: total faces detected, inference time in milliseconds, and compute backend
- Custom indigo-themed UI with a dedicated architecture/documentation panel

## Known Limitations

- Faces that are very small, heavily occluded, or turned sharply to the side may be missed, since the model works on a 300×300 resized input.
- Lowering the confidence threshold catches more faces but can also produce false positives. A threshold around 0.35–0.50 is a reasonable range to start from.

## Tech Stack

- **Python**, **NumPy**, **Pillow**
- **OpenCV** — DNN module for model loading and inference
- **Streamlit** — dashboard interface, file upload, camera input, and caching

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt

# Run the full multi-page app (model files download automatically on first load)
streamlit run Home.py
```

Navigate to the **Face Detection** page from the sidebar. The first load needs an internet connection to download the model files. Sample images are bundled in `projects/face-detection/samples/`.
