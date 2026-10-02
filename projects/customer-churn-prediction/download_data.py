import os
import sys
import shutil
import subprocess

# Directory where dataset should reside
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "Customer-Churn.csv")
ALT_CSV_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DATASET_SLUG = "blastchar/telco-customer-churn"


def download_data():
    if os.path.exists(CSV_PATH) or os.path.exists(ALT_CSV_PATH):
        print("Dataset already exists. Skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading dataset '{DATASET_SLUG}' using Kaggle CLI...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", DATA_DIR, "--unzip"],
        check=True,
    )

    # Standardize filename if downloaded with default Kaggle name
    if os.path.exists(ALT_CSV_PATH) and not os.path.exists(CSV_PATH):
        shutil.copy(ALT_CSV_PATH, CSV_PATH)

    print(f"Dataset downloaded and ready at {DATA_DIR}")


if __name__ == "__main__":
    download_data()