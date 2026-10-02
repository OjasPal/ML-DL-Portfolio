import os
import sys
import subprocess

# Directory where dataset should reside
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "emails.csv")
DATASET_SLUG = "venky73/spam-mails-dataset"


def download_data():
    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists at {CSV_PATH}. Skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading dataset '{DATASET_SLUG}' using Kaggle CLI...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", DATA_DIR, "--unzip"],
        check=True,
    )
    print(f"Dataset downloaded and extracted to {DATA_DIR}")


if __name__ == "__main__":
    download_data()