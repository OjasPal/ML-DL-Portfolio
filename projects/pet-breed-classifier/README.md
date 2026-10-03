# 🐾 Pet Breed Classifier

An interactive image classification dashboard built with Streamlit, identifying 46 cat and dog breeds from a photo using a fine-tuned EfficientNet-B0 deep learning model.

## Overview

Upload a photo of a cat or dog, or pick one of the built-in samples, and the dashboard predicts the breed, shows a confidence score and a Top-5 probability chart, and flags predictions that fall below a confidence threshold you control. The interface is built as a custom two-panel dashboard: a classification panel on the left, and a live architecture/documentation panel on the right.

## Dataset

The model is trained on a combination of two public datasets, merged into a single 46-class set (12 cat breeds and 34 dog breeds, 8,847 images in total):

- **[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)** — 37 breeds (12 cats, 25 dogs)
- **[Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)** — used to add 9 popular dog breeds missing from Oxford-IIIT Pet: German Shepherd, Labrador Retriever, Golden Retriever, Siberian Husky, Rottweiler, Doberman, French Bulldog, Pembroke Welsh Corgi, and Border Collie

The included `download_data.py` script builds this combined dataset automatically, with no Kaggle account or API token needed. See [Run Locally](#run-locally).

## Model & Architecture

- **Backbone:** EfficientNet-B0 (torchvision) with its final layer replaced by a 46-way classification head.
- **Training:** fine-tuned in two stages — first the classifier head alone, then the full network — reaching 93.6% validation accuracy.
- **Preprocessing:** `Resize(256)` → `CenterCrop(224)` → ImageNet normalization, applied identically at training and inference time. Photos from phones are auto-rotated using their EXIF orientation.
- **Prediction:** softmax over all 46 breeds, with the Top-5 shown in a chart. The cat/dog badge is derived from the predicted breed.
- **Pre-compiled artifact:** the fine-tuned weights, class names, and normalization statistics are bundled in a single file (`pet_breed_classifier.pth`), exported once from the notebook and loaded at startup with `@st.cache_resource` — the model is never retrained or rebuilt on a rerun. The artifact also stores the held-out test accuracy, which the dashboard displays when available.
- **Compute backend:** runs on the GPU (CUDA) when available and falls back to CPU otherwise. Image decoding and predictions are cached with `@st.cache_data`.

## Features

- Two input modes: image upload (JPG, JPEG, PNG) or four bundled sample images (two cats, two dogs)
- Predicted breed card with confidence percentage and a cat/dog species badge
- Top-5 prediction chart
- Adjustable confidence threshold, with a warning when a prediction falls below it
- Live metrics: model, validation accuracy, test accuracy, inference time, and compute backend
- Custom deep-teal themed UI with a dedicated architecture/documentation panel

## Known Limitations

- **Closed set of 46 breeds:** the model always picks from its 46 known classes, so a photo of an unsupported breed, a mixed breed, or a non-pet image will still be assigned to the closest of those breeds. The confidence threshold is there to flag these cases.
- **Photo quality matters:** clear, well-lit photos where the pet's face is visible give the best results.

## Tech Stack

- **Python**, **Pandas**, **Pillow**
- **PyTorch** and **torchvision** — model architecture, fine-tuning, and inference
- **Plotly** — Top-5 prediction chart
- **Streamlit** — dashboard interface and caching

## Run Locally

From the repo root:

```bash
# One-time setup
pip install -r requirements.txt

# Run the full multi-page app
streamlit run Home.py
```

The dashboard needs only the trained artifact `pet_breed_classifier.pth` in `projects/pet-breed-classifier/models/` (exported from the project notebook) and the sample images in `projects/pet-breed-classifier/samples/`. It does not need the training dataset to run. Navigate to the **Pet Breed Classifier** page from the sidebar.

### Rebuilding the training dataset

To retrain the model or reproduce the dataset, run the download script:

```bash
python projects/pet-breed-classifier/download_data.py
```

It downloads Oxford-IIIT Pet through torchvision, downloads Stanford Dogs (a large file of roughly 780 MB) and extracts only the 9 extra breeds, and combines everything into `data/pets/<Breed Name>/`, one folder per breed. If `data/pets/` already exists, the script skips the download. Once it has finished, the intermediate `data/oxford-iiit-pet/` and `data/stanford-dogs/` folders can be deleted to reclaim disk space.
