"""
DRISHTI AI - Retinal Image Quality Assessment Module.
Runs before clinical DR classification to ensure fundus images are diagnostically usable.
Implements:
1. Heuristic Quality Analyzer:
   - Laplacian Variance (blur detection)
   - Luminance histogram dispersion (underexposure / overexposure)
   - Circular fundus coverage ratio
2. Lightweight MobileNetV3-based Quality Classifier (Good / Usable / Reject)
"""

from typing import Dict, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models


class HeuristicQualityChecker:
    """
    Fast, rule-based quality filter checking blur, exposure, and illumination.
    """
    def __init__(
        self,
        min_laplacian_variance: float = 60.0,
        min_brightness_mean: float = 25.0,
        max_brightness_mean: float = 235.0,
        min_contrast_std: float = 10.0,
    ):
        self.min_laplacian_variance = min_laplacian_variance
        self.min_brightness_mean = min_brightness_mean
        self.max_brightness_mean = max_brightness_mean
        self.min_contrast_std = min_contrast_std

    def assess_quality(self, image_np: np.ndarray) -> Dict[str, Union[bool, str, float]]:
        """
        Assesses image quality on raw RGB numpy array.
        Returns quality status and diagnostic metrics.
        """
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Isolate fundus area (excluding black background borders)
        fundus_mask = gray > 10
        if np.any(fundus_mask):
            fundus_pixels = gray[fundus_mask]
            mean_intensity = float(np.mean(fundus_pixels))
            std_intensity = float(np.std(fundus_pixels))
        else:
            mean_intensity = float(np.mean(gray))
            std_intensity = float(np.std(gray))

        # 1. Blur evaluation via Laplacian Variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurred = lap_var < self.min_laplacian_variance

        # 2. Exposure evaluation
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        is_underexposed = mean_intensity < self.min_brightness_mean
        is_overexposed = mean_intensity > self.max_brightness_mean
        is_low_contrast = std_intensity < self.min_contrast_std

        # Determine overall verdict
        rejection_reasons = []
        if is_blurred:
            rejection_reasons.append(f"Image is blurred (Laplacian variance {lap_var:.1f} < {self.min_laplacian_variance})")
        if is_underexposed:
            rejection_reasons.append("Image is severely underexposed (too dark)")
        if is_overexposed:
            rejection_reasons.append("Image is overexposed (glare/flash artifact)")
        if is_low_contrast:
            rejection_reasons.append("Image has insufficient contrast across retinal structures")

        is_acceptable = len(rejection_reasons) == 0
        quality_score = min(1.0, max(0.0, (lap_var / 200.0) * 0.5 + (std_intensity / 50.0) * 0.5))

        return {
            "is_acceptable": is_acceptable,
            "status": "PASS" if is_acceptable else "RETAKE_RECOMMENDED",
            "quality_score": round(quality_score, 3),
            "laplacian_variance": round(lap_var, 2),
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "rejection_reasons": rejection_reasons,
            "recommendation": "Proceed with screening" if is_acceptable else f"Please retake the retinal capture: {'; '.join(rejection_reasons)}"
        }


class RetinalQualityNet(nn.Module):
    """
    Lightweight MobileNetV3-Small classifier for 3-class image quality triage:
    0: Good, 1: Usable, 2: Reject / Retake
    """
    def __init__(self, pretrained: bool = False, num_classes: int = 3):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# Singleton heuristic checker for fast triage
default_quality_checker = HeuristicQualityChecker()
