# ✍️ Handwritten Digit Recognizer

An interactive neural network dashboard built with Streamlit, classifying hand-drawn digits in real time using a fully connected network trained on MNIST.

## Overview

Draw a digit directly on an in-browser canvas, and a neural network predicts which digit (0–9) was drawn, along with a confidence score and the full probability distribution across all 10 classes. This is the only deep learning (neural network) project in the portfolio — the rest use classical machine learning approaches. The interface is built as a custom two-panel dashboard: a drawing and prediction panel on the left, and a live architecture/documentation panel on the right.

## Dataset

**[MNIST Handwritten Digits](https://keras.io/api/datasets/mnist/)** — 60,000 training and 10,000 test images, each 28×28 pixels grayscale. Loaded directly via `keras.datasets.mnist`, requiring no manual download or Kaggle setup (a CSV-format Kaggle mirror is also available if preferred).

## Model & Architecture

- **Architecture:** fully connected Multi-Layer Perceptron — `Dense(64, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`, taking a flattened 784-pixel input.
- **Preprocessing:** pixel values normalized to [0, 1]; canvas drawings are converted to grayscale, resized to 28×28, and flattened to match the model's expected input format.
- **Training:** Adam optimizer, sparse categorical crossentropy loss, 5 epochs.
- **Pre-compiled artifact:** the trained model (`mnist_model.keras`) is loaded directly at startup for instant inference. If no pre-compiled artifact is found, the app transparently trains the network on the fly and caches the result locally for subsequent runs.

## Features

- Live drawing canvas with real-time 28×28 preprocessed preview
- Full softmax probability distribution displayed as a bar chart across all 10 digits, not just the top prediction
- Confidence score shown alongside the predicted digit
- Custom cyan-themed UI with a dedicated architecture/documentation panel

## Known Limitations

- **Occasional misclassification:** digits drawn off-center, too small, or with unusual stroke thickness can be misread. MNIST's training images are centered by center-of-mass and anti-aliased in a specific way that a canvas drawing doesn't automatically replicate — a known gap between the training distribution and live user input.
- **Single digits only:** the model recognizes one digit per prediction. Multi-digit numbers are not supported, since MNIST itself only contains isolated single-digit images — recognizing full numbers would require a separate digit-segmentation step before classification.

## Tech Stack

- **Python**, **NumPy**, **Pillow**
- **Keras** / **TensorFlow** — model architecture and training
- **Streamlit** + **streamlit-drawable-canvas** — interactive drawing interface and caching

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt

# Run the full multi-page app (no dataset download needed — MNIST loads via Keras)
streamlit run Home.py
```

Navigate to the **Handwritten Digit Recognizer** page from the sidebar.
