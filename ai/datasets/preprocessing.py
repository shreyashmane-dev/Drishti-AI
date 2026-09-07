"""
DRISHTI AI - Clinical Preprocessing Pipeline for Fundus Photography.
Includes:
- Circular boundary cropping
- Ben Graham illumination normalization (Gaussian-blur background subtraction)
- CLAHE (Contrast Limited Adaptive Histogram Equalization) on green channel
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def crop_fundus_circular(img: np.ndarray, tolerance: int = 10) -> np.ndarray:
    """
    Crops black borders and isolates the circular fundus region.
    Args:
        img: RGB or BGR numpy array image.
        tolerance: pixel intensity threshold for background detection.
    Returns:
        Cropped numpy array image.
    """
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    mask = gray > tolerance
    
    # Check if image is completely dark
    if not np.any(mask):
        return img
        
    # Get bounding box of non-dark pixels
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    cropped = img[y0:y1, x0:x1]
    return cropped


def ben_graham_normalization(
    img: np.ndarray,
    sigma: int = 30,
    alpha: float = 4.0,
    beta: float = -4.0,
    gamma: float = 128.0,
) -> np.ndarray:
    """
    Applies Ben Graham's local illumination normalization via Gaussian-blur subtraction:
    I_norm = alpha * I + beta * GaussianBlur(I, sigma) + gamma
    This standardizes diverse lighting conditions across fundus cameras.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    normalized = cv2.addWeighted(img, alpha, blurred, beta, gamma)
    return normalized


def apply_green_clahe(
    img: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE to the Green channel of the RGB image.
    The green channel has the highest spectral absorption for retinal microaneurysms,
    hemorrhages, and hard/soft exudates.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    r, g, b = cv2.split(img)
    g_enhanced = clahe.apply(g)
    merged = cv2.merge([r, g_enhanced, b])
    return merged


def preprocess_fundus_image(
    image_input,
    target_size: tuple = (512, 512),
    apply_bg: bool = True,
    apply_clahe: bool = True,
    sigma: int = 30,
) -> np.ndarray:
    """
    Complete clinical preprocessing pipeline:
    1. Convert input to RGB numpy array
    2. Circular fundus crop
    3. Resize to target dimensions
    4. Ben Graham Gaussian subtraction (illumination invariance)
    5. Green-channel CLAHE
    Returns:
        RGB numpy array (uint8, target_size)
    """
    if isinstance(image_input, (str, bytes, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not read image from {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif hasattr(image_input, "__fspath__"):
        img = cv2.imread(str(image_input.__fspath__()))
        if img is None:
            raise ValueError(f"Could not read image from {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 1. Circular boundary crop
    img_cropped = crop_fundus_circular(img)
    
    # 2. Resize
    img_resized = cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_AREA)

    # 3. Ben Graham Normalization
    if apply_bg:
        img_norm = ben_graham_normalization(img_resized, sigma=sigma)
    else:
        img_norm = img_resized

    # 4. CLAHE on Green channel
    if apply_clahe:
        img_final = apply_green_clahe(img_norm)
    else:
        img_final = img_norm

    return np.clip(img_final, 0, 255).astype(np.uint8)


def get_inference_tensor(
    image_input,
    target_size: tuple = (512, 512),
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    device: str = "cpu"
) -> torch.Tensor:
    """
    Transforms raw image input into standard PyTorch tensor ready for model inference.
    """
    processed_np = preprocess_fundus_image(image_input, target_size=target_size)
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),  # Scales to [0.0, 1.0]
        transforms.Normalize(mean=mean, std=std)
    ])
    tensor = tensor_transform(processed_np).unsqueeze(0).to(device)
    return tensor
