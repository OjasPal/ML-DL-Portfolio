import os
import subprocess

# Kaggle dataset slug: ruchi798/housing-prices-in-metropolitan-areas-of-india
DATASET_SLUG = "ruchi798/housing-prices-in-metropolitan-areas-of-india"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TARGET_FILE = os.path.join(DATA_DIR, "Delhi.csv")


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TARGET_FILE):
        print(f"File already exists at: {TARGET_FILE}. Skipping download.")
        return

    print(f"Downloading {DATASET_SLUG} from Kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", DATA_DIR, "--unzip"],
            check=True,
        )
        print(f"Successfully downloaded and extracted dataset to {DATA_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset. Ensure KAGGLE_API_TOKEN is set. Error: {e}")


if __name__ == "__main__":
    download_data()