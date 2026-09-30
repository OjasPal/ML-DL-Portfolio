import os
import subprocess

# Paths setup
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "spotify_songs.csv")
DATASET_SLUG = "joebeachcapital/30000-spotify-songs"


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if dataset already exists
    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists at: {CSV_PATH}")
        print("Skipping download.")
        return

    print(f"Downloading dataset '{DATASET_SLUG}' using Kaggle CLI...")

    import sys

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        DATA_DIR,
        "--unzip",
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("Download and extraction completed successfully!")
    else:
        print(f"Failed to download dataset. Kaggle CLI exited with code {result.returncode}")


if __name__ == "__main__":
    download_data()