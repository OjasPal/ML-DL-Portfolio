import os
import shutil
import tarfile
import urllib.request

from torchvision import datasets

# Directory where the dataset should reside
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PETS_DIR = os.path.join(DATA_DIR, "pets")

OXFORD_IMAGES = os.path.join(DATA_DIR, "oxford-iiit-pet", "images")
STANFORD_DIR = os.path.join(DATA_DIR, "stanford-dogs")
STANFORD_IMAGES = os.path.join(STANFORD_DIR, "Images")
STANFORD_TAR = os.path.join(STANFORD_DIR, "images.tar")
STANFORD_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

# Famous breeds missing from Oxford-IIIT Pet (names as in the Stanford Dogs folders)
EXTRA_BREEDS = ["German_shepherd", "Labrador_retriever", "golden_retriever", "Siberian_husky",
                "Rottweiler", "Doberman", "French_bulldog", "Pembroke", "Border_collie"]


def clean_name(name):
    return name.replace("_", " ").title()


def download_data():
    if os.path.exists(PETS_DIR):
        print(f"Dataset already exists at {PETS_DIR}. Skipping download.")
        return

    # 1. Oxford-IIIT Pet (37 breeds) via torchvision
    print("Downloading Oxford-IIIT Pet...")
    datasets.OxfordIIITPet(root=DATA_DIR, split="trainval", download=True)

    # 2. Stanford Dogs (only needed for the extra breeds)
    if not os.path.exists(STANFORD_IMAGES):
        os.makedirs(STANFORD_DIR, exist_ok=True)
        if not os.path.exists(STANFORD_TAR):
            print("Downloading Stanford Dogs (~780 MB)...")
            urllib.request.urlretrieve(STANFORD_URL, STANFORD_TAR)
        print("Extracting Stanford Dogs...")
        with tarfile.open(STANFORD_TAR) as tar:
            tar.extractall(STANFORD_DIR)

    # 3. Combine both into data/pets/<Breed Name>/
    for file in os.listdir(OXFORD_IMAGES):          # e.g. "american_bulldog_10.jpg"
        if file.endswith(".jpg"):
            breed_dir = os.path.join(PETS_DIR, clean_name(file.rsplit("_", 1)[0]))
            os.makedirs(breed_dir, exist_ok=True)
            shutil.copy2(os.path.join(OXFORD_IMAGES, file), breed_dir)

    extra = {b.lower() for b in EXTRA_BREEDS}
    added = 0
    for folder in os.listdir(STANFORD_IMAGES):      # e.g. "n02106662-German_shepherd"
        breed = folder.split("-", 1)[1]
        if breed.lower() in extra:
            shutil.copytree(os.path.join(STANFORD_IMAGES, folder), os.path.join(PETS_DIR, clean_name(breed)))
            added += 1

    if added != len(EXTRA_BREEDS):
        print(f"Warning: only {added} of {len(EXTRA_BREEDS)} extra breeds were found.")
    print(f"Dataset ready at {PETS_DIR} ({len(os.listdir(PETS_DIR))} breeds)")


if __name__ == "__main__":
    download_data()
