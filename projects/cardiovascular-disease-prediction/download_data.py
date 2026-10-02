import os
import sys
import subprocess
import pandas as pd

# Directory where dataset should reside
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_RAW = os.path.join(DATA_DIR, "cardio_train.csv")
CSV_CLEANED = os.path.join(DATA_DIR, "cardio_train_cleaned.csv")
DATASET_SLUG = "sulianova/cardiovascular-disease-dataset"


def download_data():
    if os.path.exists(CSV_CLEANED) or os.path.exists(CSV_RAW):
        print("Dataset already exists. Skipping download.")
        # Ensure cleaned file exists if raw exists
        if os.path.exists(CSV_RAW) and not os.path.exists(CSV_CLEANED):
            df = pd.read_csv(CSV_RAW, sep=";")
            df.to_csv(CSV_CLEANED, index=False)
            print("Cleaned CSV file generated successfully.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading dataset '{DATASET_SLUG}' using Kaggle CLI...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", DATA_DIR, "--unzip"],
        check=True,
    )

    # Convert semicolon-separated CSV to standard comma-separated CSV matching notebook
    if os.path.exists(CSV_RAW):
        df = pd.read_csv(CSV_RAW, sep=";")
        df.to_csv(CSV_CLEANED, index=False)
        print("Cleaned CSV file generated successfully.")

    print(f"Dataset downloaded and ready at {DATA_DIR}")


if __name__ == "__main__":
    download_data()