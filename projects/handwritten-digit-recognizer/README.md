# ✍️ Handwritten Digit Recognizer

A neural network that classifies hand-drawn digits (0-9) in real time — deployed as an interactive Streamlit page within the [ML-DL-Portfolio](../../) app.

## Overview

Draw a digit directly on the canvas, and a neural network trained on the MNIST dataset predicts what you drew, along with its confidence across all 10 possible digits.

This is the only deep learning (neural network) project in the portfolio — the rest use classical machine learning approaches (regression, clustering, similarity-based recommenders).

## Dataset

**[MNIST Handwritten Digits](https://keras.io/api/datasets/mnist/)** — 60,000 training images and 10,000 test images of handwritten digits, each 28×28 pixels grayscale. Bundled directly with Keras (`keras.datasets.mnist`), so no manual download or Kaggle API setup is needed for this project.

## Model

- **Architecture:** Simple feedforward neural network (Keras `Sequential`)
  - Dense layer, 64 units, ReLU activation
  - Dense layer, 64 units, ReLU activation
  - Dense output layer, 10 units, Softmax activation
- **Preprocessing:** pixel values normalized to [0, 1], images flattened from 28×28 to a 784-length vector
- **Training:** Adam optimizer, sparse categorical crossentropy loss, 5 epochs
- **Accuracy:** ~98% on training data

## How It Works

1. User draws a single digit on an in-browser canvas
2. The drawing is converted to grayscale, resized to 28×28, and normalized to match the model's expected input format
3. The trained model predicts probabilities across all 10 digit classes
4. The highest-probability digit is shown as the prediction, along with a confidence percentage and a bar chart of all 10 class probabilities

## Tech Stack

- **Python**, **NumPy**, **Pillow**
- **Keras** / **TensorFlow** — model architecture and training
- **Streamlit** + **streamlit-drawable-canvas** — interactive drawing UI

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt

# Run the full multi-page app (no separate data download needed — MNIST loads via Keras)
streamlit run Home.py
```

Then navigate to the **Handwritten Digit Recognizer** page from the sidebar.

## Known Limitations

- **Occasional misclassification:** digits drawn off-center, too small, or with unusual stroke thickness can be misread (e.g. a "1" or "6" misclassified as "8"). This happens because MNIST's training images are centered by center-of-mass and anti-aliased in a specific way that a canvas drawing doesn't automatically replicate — a known real-world gap between the training distribution and live user input.
- **Single digits only:** the model only recognizes one digit per prediction. Multi-digit numbers (e.g. "16") are not supported, since MNIST itself only contains isolated single-digit images — recognizing full numbers would require a separate digit-segmentation step before classification.

## Notes

Originally a short practice notebook exploring a basic neural network on MNIST. Expanded here with a live drawing interface to make the deep learning concept tangible and interactive, rather than just printing a static test-set prediction.
