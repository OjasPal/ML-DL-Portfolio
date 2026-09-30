import os
import subprocess
import zipfile
import pandas as pd

DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
DATA_DIR = "data"
TRIMMED_ROWS = 50_000
FINAL_FILENAME = "tmdb_movies_50k.csv"

def download_full_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    final_path = os.path.join(DATA_DIR, FINAL_FILENAME)
    if os.path.exists(final_path):
        print(f"Trimmed dataset already exists at '{final_path}'. Skipping.")
        return None

    print(f"Downloading '{DATASET}' from Kaggle (this is the full ~650MB+ file)...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR],
        check=True
    )

    for file in os.listdir(DATA_DIR):
        if file.endswith(".zip"):
            zip_path = os.path.join(DATA_DIR, file)
            print(f"Extracting {file}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(zip_path)

    # Find the extracted CSV (name may vary slightly)
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f != FINAL_FILENAME]
    if not csv_files:
        raise FileNotFoundError("No CSV found after extraction. Check the dataset contents.")
    return os.path.join(DATA_DIR, csv_files[0])

def trim_dataset(full_csv_path):
    print(f"Loading full dataset from '{full_csv_path}'... (this may take a minute, it's a big file)")
    df = pd.read_csv(full_csv_path, low_memory=False)
    print(f"Full dataset shape: {df.shape}")

    # Keep the most-rated / most-relevant movies instead of a random slice
    df_trimmed = df.sort_values("vote_count", ascending=False).head(TRIMMED_ROWS)

    final_path = os.path.join(DATA_DIR, FINAL_FILENAME)
    df_trimmed.to_csv(final_path, index=False)
    print(f"Saved trimmed dataset ({df_trimmed.shape[0]} rows) to '{final_path}'")

    # Delete the huge original file to save disk space
    os.remove(full_csv_path)
    print(f"Deleted full-size file '{full_csv_path}' to save space.")

if __name__ == "__main__":
    full_csv = download_full_dataset()
    if full_csv:
        trim_dataset(full_csv)
