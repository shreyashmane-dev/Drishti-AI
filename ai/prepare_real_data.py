"""
DRISHTI AI - Real Dataset Indexer & Format Standardizer.
Indexes the 3,662 real clinical fundus images into unified tabular format.
"""

from pathlib import Path
import shutil
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import DATA_DIR


def organize_aptos_real_dataset():
    """Indexes the downloaded APTOS 2019 dataset."""
    base_path = DATA_DIR / "aptos2019" / "gaussian_filtered_images" / "gaussian_filtered_images"
    dest_images_dir = DATA_DIR / "aptos2019" / "real_images"
    dest_images_dir.mkdir(parents=True, exist_ok=True)

    folder_to_grade = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferate_DR": 4,
    }

    records = []
    print("📁 Indexing real clinical fundus images from APTOS 2019...")

    for folder_name, grade in folder_to_grade.items():
        src_folder = base_path / folder_name
        if not src_folder.exists():
            continue
            
        img_files = list(src_folder.glob("*.png")) + list(src_folder.glob("*.jpg"))
        for img_path in img_files:
            dest_file = dest_images_dir / img_path.name
            # Copy or symlink image if needed
            if not dest_file.exists():
                shutil.copy2(img_path, dest_file)

            records.append({
                "image_id": img_path.stem,
                "diagnosis": grade,
                "grade_name": folder_name,
                "file_path": str(dest_file.relative_to(DATA_DIR)),
            })

    df = pd.DataFrame(records)
    csv_out = DATA_DIR / "aptos2019" / "real_train.csv"
    df.to_csv(csv_out, index=False)

    print(f"✅ Successfully indexed {len(df)} real images into: {csv_out}")
    print(df["grade_name"].value_counts())
    return df


if __name__ == "__main__":
    organize_aptos_real_dataset()
