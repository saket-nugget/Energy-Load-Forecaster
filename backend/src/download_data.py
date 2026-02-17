import kagglehub
import shutil
import os

def download_energy_data():
    # Download dataset via kagglehub
    path = kagglehub.dataset_download("raminhuseyn/energy-consumption-dataset")
    print("Downloaded dataset to:", path)

    # Ensure data/raw folder exists
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Move all CSV files from cache to data/raw
    for file in os.listdir(path):
        if file.endswith(".csv"):
            src = os.path.join(path, file)
            dst = os.path.join(raw_dir, file)
            shutil.copy(src, dst)
            print(f"Copied {file} → {dst}")

if __name__ == "__main__":
    download_energy_data()
