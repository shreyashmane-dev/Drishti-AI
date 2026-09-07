"""
DRISHTI AI - Data Augmentation Pipeline for Diabetic Retinopathy.
Uses Albumentations / torchvision with clinical constraints:
- 360-degree rotation (fundus images have no canonical orientation)
- Flips & slight affine transforms
- Subtle illumination jitter (avoiding aggressive color/hue shifts that corrupt diagnostic features)
"""

from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transforms(
    image_size: Tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Returns data augmentation pipeline for training.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # 360-degree arbitrary rotation
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5
        ),
        # Mild brightness & contrast jitter (safe for clinical diagnosis)
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        # Subtle Gaussian blur for sensor noise robustness
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_validation_transforms(
    image_size: Tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Returns deterministic validation/testing transformation pipeline.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
