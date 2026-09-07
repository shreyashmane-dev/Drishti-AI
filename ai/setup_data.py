"""
DRISHTI AI - Dataset Setup & Synthetic Verification Generator.
1. Generates structured sample fundus cohorts (APTOS, EyePACS, IDRiD lesion masks, Messidor-2)
   to allow immediate end-to-end testing of training, Grad-CAM, and validation pipelines.
2. Provides CLI download helpers and directory structure for full Kaggle datasets.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from ai.config import DATA_DIR


def generate_synthetic_fundus_image(
    filename: Path,
    grade: int,
    size: tuple = (512, 512),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a realistic synthetic fundus photograph with retinal structures and lesions
    correlated with the ICDR grade (0=Normal, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative).
    Returns (fundus_rgb, lesion_mask_binary).
    """
    h, w = size
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.44)

    # 1. Base orange-red retinal background with natural gradient
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    fundus_mask = dist_from_center <= radius

    # Background pigment variation
    red_channel = np.clip(180 - (dist_from_center / radius * 60) + np.random.normal(0, 5, (h, w)), 0, 255)
    green_channel = np.clip(80 - (dist_from_center / radius * 40) + np.random.normal(0, 4, (h, w)), 0, 255)
    blue_channel = np.clip(30 - (dist_from_center / radius * 20) + np.random.normal(0, 2, (h, w)), 0, 255)

    img[..., 0] = np.where(fundus_mask, red_channel, 0)
    img[..., 1] = np.where(fundus_mask, green_channel, 0)
    img[..., 2] = np.where(fundus_mask, blue_channel, 0)

    # 2. Optic Disc (Bright yellowish circle at nasal position)
    optic_disc_center = (int(center[0] - radius * 0.45), center[1])
    optic_disc_radius = int(radius * 0.16)
    cv2.circle(img, optic_disc_center, optic_disc_radius, (240, 220, 130), -1)

    # Optic cup (inner lighter zone)
    cv2.circle(img, optic_disc_center, int(optic_disc_radius * 0.5), (255, 245, 180), -1)

    # 3. Retinal Blood Vessels (Arching branches from optic disc)
    vessel_color = (110, 20, 15)
    for angle_deg in [-50, -25, 0, 25, 50, 130, 155, 180, 205, 230]:
        rad = math.radians(angle_deg)
        pts = [optic_disc_center]
        curr_x, curr_y = float(optic_disc_center[0]), float(optic_disc_center[1])
        for step in range(12):
            curr_x += math.cos(rad) * (radius * 0.08) + np.random.uniform(-3, 3)
            curr_y += math.sin(rad) * (radius * 0.08) + np.random.uniform(-3, 3)
            pts.append((int(curr_x), int(curr_y)))
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        thickness = max(1, int(4 - step * 0.2))
        cv2.polylines(img, [pts_np], isClosed=False, color=vessel_color, thickness=thickness)

    # Lesion mask for XAI validation
    lesion_mask = np.zeros((h, w), dtype=np.uint8)

    # 4. Inject lesions according to ICDR DR grade
    np.random.seed(int(sum(ord(c) for c in filename.name)) % 10000)

    # Grade 1+: Microaneurysms (tiny red dots)
    if grade >= 1:
        n_ma = np.random.randint(5, 15) if grade == 1 else np.random.randint(15, 40)
        for _ in range(n_ma):
            ma_x = int(center[0] + np.random.uniform(-radius*0.6, radius*0.6))
            ma_y = int(center[1] + np.random.uniform(-radius*0.6, radius*0.6))
            if fundus_mask[ma_y, ma_x]:
                cv2.circle(img, (ma_x, ma_y), np.random.randint(2, 4), (120, 10, 10), -1)
                cv2.circle(lesion_mask, (ma_x, ma_y), 4, 255, -1)

    # Grade 2+: Hard Exudates (bright yellowish-white lipid deposits)
    if grade >= 2:
        n_ex = np.random.randint(4, 12)
        for _ in range(n_ex):
            ex_x = int(center[0] + np.random.uniform(-radius*0.5, radius*0.5))
            ex_y = int(center[1] + np.random.uniform(-radius*0.5, radius*0.5))
            if fundus_mask[ex_y, ex_x]:
                cv2.circle(img, (ex_x, ex_y), np.random.randint(3, 7), (250, 245, 160), -1)
                cv2.circle(lesion_mask, (ex_x, ex_y), 7, 255, -1)

    # Grade 3+: Blot Haemorrhages & Cotton Wool Spots
    if grade >= 3:
        n_he = np.random.randint(8, 20)
        for _ in range(n_he):
            he_x = int(center[0] + np.random.uniform(-radius*0.6, radius*0.6))
            he_y = int(center[1] + np.random.uniform(-radius*0.6, radius*0.6))
            if fundus_mask[he_y, he_x]:
                cv2.ellipse(img, (he_x, he_y), (np.random.randint(6, 14), np.random.randint(4, 9)), 
                            np.random.randint(0, 180), 0, 360, (90, 5, 5), -1)
                cv2.ellipse(lesion_mask, (he_x, he_y), (14, 9), 0, 0, 360, 255, -1)

    # Grade 4: Neovascularization (tangle of fragile abnormal vessels)
    if grade == 4:
        for _ in range(4):
            nv_x = int(optic_disc_center[0] + np.random.uniform(-30, 30))
            nv_y = int(optic_disc_center[1] + np.random.uniform(-30, 30))
            for _ in range(6):
                cv2.circle(img, (nv_x + np.random.randint(-15, 15), nv_y + np.random.randint(-15, 15)), 2, (140, 20, 20), -1)
                cv2.circle(lesion_mask, (nv_x, nv_y), 15, 255, -1)

    # Add subtle lens blur and sensor noise
    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    # Save to disk
    filename.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return img, lesion_mask


def create_sample_cohorts():
    """Generates sample datasets for APTOS, EyePACS, IDRiD, and Messidor-2."""
    print("📦 Creating sample verification cohorts in data/ directory...")

    # 1. APTOS 2019 Sample Dataset
    aptos_dir = DATA_DIR / "aptos2019"
    aptos_images_dir = aptos_dir / "train_images"
    aptos_images_dir.mkdir(parents=True, exist_ok=True)

    aptos_records = []
    # Generate 50 balanced sample images across grades 0-4
    for grade in range(5):
        for i in range(10):
            img_id = f"aptos_sample_g{grade}_{i:02d}"
            img_path = aptos_images_dir / f"{img_id}.png"
            generate_synthetic_fundus_image(img_path, grade=grade)
            aptos_records.append({"id_code": img_id, "diagnosis": grade})

    aptos_df = pd.DataFrame(aptos_records)
    aptos_df.to_csv(aptos_dir / "train.csv", index=False)
    print(f"   ✓ Generated APTOS 2019 sample dataset: {len(aptos_records)} images at {aptos_dir}")

    # 2. EyePACS Sample Dataset
    eyepacs_dir = DATA_DIR / "eyepacs"
    eyepacs_images_dir = eyepacs_dir / "train_images"
    eyepacs_images_dir.mkdir(parents=True, exist_ok=True)

    eyepacs_records = []
    for grade in range(5):
        for i in range(6):
            img_id = f"eyepacs_sample_g{grade}_{i:02d}"
            img_path = eyepacs_images_dir / f"{img_id}.jpeg"
            generate_synthetic_fundus_image(img_path, grade=grade)
            eyepacs_records.append({"image": img_id, "level": grade})

    eyepacs_df = pd.DataFrame(eyepacs_records)
    eyepacs_df.to_csv(eyepacs_dir / "trainLabels.csv", index=False)
    print(f"   ✓ Generated EyePACS sample dataset: {len(eyepacs_records)} images at {eyepacs_dir}")

    # 3. IDRiD Sample Dataset with Ground Truth Lesion Masks
    idrid_dir = DATA_DIR / "idrid"
    idrid_images_dir = idrid_dir / "images"
    idrid_masks_dir = idrid_dir / "masks"
    idrid_images_dir.mkdir(parents=True, exist_ok=True)
    idrid_masks_dir.mkdir(parents=True, exist_ok=True)

    idrid_records = []
    for grade in range(5):
        for i in range(4):
            img_id = f"IDRiD_sample_g{grade}_{i:02d}"
            img_path = idrid_images_dir / f"{img_id}.jpg"
            mask_path = idrid_masks_dir / f"{img_id}_mask.png"
            _, mask = generate_synthetic_fundus_image(img_path, grade=grade)
            cv2.imwrite(str(mask_path), mask)
            idrid_records.append({"Image name": img_id, "Retinopathy grade": grade})

    idrid_df = pd.DataFrame(idrid_records)
    idrid_df.to_csv(idrid_dir / "idrid_grades.csv", index=False)
    print(f"   ✓ Generated IDRiD dataset with lesion masks: {len(idrid_records)} images at {idrid_dir}")

    # 4. Messidor-2 Sample Dataset
    messidor_dir = DATA_DIR / "messidor2"
    messidor_images_dir = messidor_dir / "images"
    messidor_images_dir.mkdir(parents=True, exist_ok=True)

    messidor_records = []
    for grade in range(5):
        for i in range(5):
            img_id = f"messidor2_sample_g{grade}_{i:02d}"
            img_path = messidor_images_dir / f"{img_id}.png"
            generate_synthetic_fundus_image(img_path, grade=grade)
            messidor_records.append({"image_id": img_id, "adjudicated_dr_grade": grade})

    messidor_df = pd.DataFrame(messidor_records)
    messidor_df.to_csv(messidor_dir / "messidor2_grades.csv", index=False)
    print(f"   ✓ Generated Messidor-2 validation dataset: {len(messidor_records)} images at {messidor_dir}")


def print_kaggle_download_instructions():
    """Prints instructions for downloading original Kaggle competition datasets."""
    print("""
================================================================================
                    KAGGLE DATASET DOWNLOAD INSTRUCTIONS
================================================================================

To train on the full research datasets, place your kaggle.json in ~/.kaggle/
and run the following commands:

1. APTOS 2019 Blindness Detection (Aravind Eye Hospital, India):
   kaggle competitions download -c aptos2019-blindness-detection -p data/aptos2019/
   unzip data/aptos2019/aptos2019-blindness-detection.zip -d data/aptos2019/

2. EyePACS Diabetic Retinopathy Detection:
   kaggle competitions download -c diabetic-retinopathy-detection -p data/eyepacs/
   
3. IDRiD (Indian Diabetic Retinopathy Image Dataset):
   kaggle datasets download -d mariaherrerot/idrid-dataset -p data/idrid/

4. Messidor-2:
   kaggle datasets download -d google-brain/messidor2-dr-grades -p data/messidor2/
================================================================================
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRISHTI AI Data Setup & Generator")
    parser.add_argument("--create_samples", action="store_true", default=True, help="Generate sample cohorts for immediate verification")
    parser.add_argument("--show_instructions", action="store_true", help="Show Kaggle download commands")
    args = parser.parse_args()

    if args.create_samples:
        create_sample_cohorts()
    if args.show_instructions:
        print_kaggle_download_instructions()
