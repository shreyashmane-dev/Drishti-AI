"""
DRISHTI AI - Automated Kaggle & Open Source Dataset Ingestion Script.
Downloads, unzips, and prepares:
1. APTOS 2019 Blindness Detection (Primary Training Cohort)
2. IDRiD (Lesion Segmentation & Explainability Ground Truth)
3. Messidor-2 (External Generalization Test Cohort)
4. EyePACS (Large Pretraining Cohort)
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import zipfile

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import DATA_DIR


def setup_kaggle_credentials(username: str = None, key: str = None):
    """Sets environment variables or verifies kaggle.json."""
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username.strip()
        os.environ["KAGGLE_KEY"] = key.strip()
        print(f"✓ Configured Kaggle credentials for user: {username}")
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(kaggle_dir / "kaggle.json", "w") as f:
            json.dump({"username": username.strip(), "key": key.strip()}, f)
    
    # Check if local project has kaggle.json
    local_kaggle = Path("kaggle.json")
    if local_kaggle.exists():
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy("kaggle.json", kaggle_dir / "kaggle.json")
        print("✓ Copied local kaggle.json to ~/.kaggle/kaggle.json")


def extract_zip(zip_path: Path, target_dir: Path):
    """Extracts a zip file into target directory."""
    print(f"📦 Extracting {zip_path.name} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"✓ Extracted successfully.")


def download_aptos2019(target_dir: Path = None):
    """Downloads and unzips APTOS 2019 Blindness Detection dataset."""
    target_dir = target_dir or (DATA_DIR / "aptos2019")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n⬇️ [1/4] Downloading APTOS 2019 Blindness Detection (~3,662 images)...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("aptos2019-blindness-detection", path=str(target_dir))
        
        # Check downloaded zip
        zip_candidates = list(target_dir.glob("*.zip"))
        for z in zip_candidates:
            extract_zip(z, target_dir)
            z.unlink()  # Remove zip after extraction to save disk space
        print("✅ APTOS 2019 dataset ready in:", target_dir)
    except Exception as e:
        print(f"❌ Error downloading APTOS 2019: {e}")


def download_idrid(target_dir: Path = None):
    """Downloads IDRiD dataset (Images + Ground-truth Lesion Masks)."""
    target_dir = target_dir or (DATA_DIR / "idrid")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n⬇️ [2/4] Downloading IDRiD Dataset (Indian population with lesion masks)...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("mariaherrerot/idrid-dataset", path=str(target_dir), unzip=True)
        print("✅ IDRiD dataset ready in:", target_dir)
    except Exception as e:
        print(f"❌ Error downloading IDRiD: {e}")


def download_messidor2(target_dir: Path = None):
    """Downloads Messidor-2 external validation dataset."""
    target_dir = target_dir or (DATA_DIR / "messidor2")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n⬇️ [3/4] Downloading Messidor-2 Dataset (External validation)...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("google-brain/messidor2-dr-grades", path=str(target_dir), unzip=True)
        print("✅ Messidor-2 dataset ready in:", target_dir)
    except Exception as e:
        print(f"❌ Error downloading Messidor-2: {e}")


def main():
    parser = argparse.ArgumentParser(description="DRISHTI AI Dataset Downloader")
    parser.add_argument("--username", type=str, default=None, help="Kaggle Username")
    parser.add_argument("--key", type=str, default=None, help="Kaggle API Key")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "aptos", "idrid", "messidor"], help="Which dataset to download")
    args = parser.parse_args()

    setup_kaggle_credentials(username=args.username, key=args.key)

    if args.dataset in ["all", "aptos"]:
        download_aptos2019()
    if args.dataset in ["all", "idrid"]:
        download_idrid()
    if args.dataset in ["all", "messidor"]:
        download_messidor2()


if __name__ == "__main__":
    main()
