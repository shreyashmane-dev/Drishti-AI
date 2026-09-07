"""
DRISHTI AI - Unified Dataset Loader for Multi-Source Fundus Cohorts.
Unifies:
1. APTOS 2019 Blindness Detection (Aravind Eye Hospital, India)
2. EyePACS Diabetic Retinopathy (Kaggle 2015)
3. Messidor-2 (External French validation cohort)
4. IDRiD (Indian Diabetic Retinopathy Image Dataset with Lesion Masks)
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from ai.datasets.preprocessing import preprocess_fundus_image


class UnifiedFundusDataset(Dataset):
    """
    Unified PyTorch Dataset supporting APTOS, EyePACS, Messidor-2, IDRiD,
    and custom tabular/folder representations.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Union[str, Path],
        image_col: str = "image_id",
        label_col: str = "diagnosis",
        transform: Optional[Callable] = None,
        apply_ben_graham: bool = True,
        apply_clahe: bool = True,
        target_size: Tuple[int, int] = (512, 512),
        image_extension: str = "",
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.apply_ben_graham = apply_ben_graham
        self.apply_clahe = apply_clahe
        self.target_size = target_size
        self.image_extension = image_extension

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_name: str) -> Path:
        """Finds image file matching name with common fundus extensions."""
        candidate = self.images_dir / raw_name
        if candidate.exists():
            return candidate
            
        if self.image_extension and not raw_name.endswith(self.image_extension):
            cand_ext = self.images_dir / f"{raw_name}{self.image_extension}"
            if cand_ext.exists():
                return cand_ext

        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]:
            cand = self.images_dir / f"{raw_name}{ext}"
            if cand.exists():
                return cand

        raise FileNotFoundError(f"Image not found for ID '{raw_name}' in '{self.images_dir}'")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = str(row[self.image_col])
        label = int(row[self.label_col]) if self.label_col in row and pd.notna(row[self.label_col]) else -1

        img_path = self._resolve_image_path(image_id)
        
        # Clinical preprocessing pipeline
        processed_img = preprocess_fundus_image(
            img_path,
            target_size=self.target_size,
            apply_bg=self.apply_ben_graham,
            apply_clahe=self.apply_clahe,
        )

        if self.transform:
            augmented = self.transform(image=processed_img)
            tensor_img = augmented["image"]
        else:
            tensor_img = torch.tensor(processed_img).permute(2, 0, 1).float() / 255.0

        return tensor_img, label, image_id


def load_aptos_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads APTOS 2019 dataset annotations (id_code, diagnosis)."""
    df = pd.read_csv(csv_path)
    if "id_code" in df.columns:
        df = df.rename(columns={"id_code": "image_id"})
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(int)
    df["source"] = "APTOS_2019"
    return df


def load_eyepacs_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads EyePACS Kaggle 2015 annotations (image, level)."""
    df = pd.read_csv(csv_path)
    rename_dict = {}
    if "image" in df.columns:
        rename_dict["image"] = "image_id"
    if "level" in df.columns:
        rename_dict["level"] = "diagnosis"
    df = df.rename(columns=rename_dict)
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["source"] = "EyePACS"
    return df


def load_messidor2_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads Messidor-2 dataset annotations."""
    df = pd.read_csv(csv_path)
    rename_dict = {}
    for col in df.columns:
        lower = col.lower()
        if "image" in lower:
            rename_dict[col] = "image_id"
        elif "grade" in lower or "diagnosis" in lower or "dr" in lower:
            rename_dict[col] = "diagnosis"
    df = df.rename(columns=rename_dict)
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(int)
    df["source"] = "Messidor_2"
    return df


def load_idrid_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads IDRiD clinical grade annotations."""
    df = pd.read_csv(csv_path)
    rename_dict = {}
    for col in df.columns:
        lower = col.lower()
        if "image" in lower or "name" in lower:
            rename_dict[col] = "image_id"
        elif "retinopathy" in lower or "grade" in lower:
            rename_dict[col] = "diagnosis"
    df = df.rename(columns=rename_dict)
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(int)
    df["source"] = "IDRiD"
    return df
