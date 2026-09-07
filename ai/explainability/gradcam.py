"""
DRISHTI AI - Explainable AI: Gradient-Weighted Class Activation Mapping (Grad-CAM).
Provides transparent visual evidence by localizing retinal biomarkers (microaneurysms,
haemorrhages, exudates, neovascularization) that drive the model's clinical prediction.
"""

import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM engine for DrishtiRetinopathyNet.
    Hooks into the final convolutional feature extractor stage.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # If target layer not specified, automatically find final conv layer
        if target_layer is None:
            if hasattr(model, "target_conv_layer"):
                self.target_layer = model.target_conv_layer
            elif hasattr(model, "features"):
                self.target_layer = model.features[-1]
            else:
                raise ValueError("Could not automatically locate target convolutional layer.")
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap for the specified target class.
        Args:
            input_tensor: (1, 3, H, W) normalized image tensor.
            target_class: ICDR class index (0-4). If None, uses predicted class.
        Returns:
            2D numpy array with values in [0.0, 1.0].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass on the score of the target class
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_class] = 1.0
        output.backward(gradient=one_hot_output, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")

        # Global average pooling on gradients -> channel weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)

        # Apply ReLU to retain only positive influences on target class
        cam = F.relu(cam)

        # Upsample to match input image resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization with numerical stability epsilon
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_gradcam(
    original_image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays 2D Grad-CAM heatmap on RGB image.
    Args:
        original_image_rgb: (H, W, 3) uint8 RGB image
        heatmap: (H, W) float heatmap in [0.0, 1.0]
        alpha: heatmap transparency ratio (0.0 to 1.0)
    Returns:
        (H, W, 3) uint8 RGB blended visualization
    """
    h, w = original_image_rgb.shape[:2]
    
    # Resize heatmap if shapes differ
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    # Convert heatmap to 8-bit colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Alpha blending: I_overlay = (1 - alpha) * I_orig + alpha * I_cam
    blended = cv2.addWeighted(original_image_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return blended


def encode_image_to_base64(image_rgb: np.ndarray, format: str = "PNG") -> str:
    """Encodes an RGB numpy array to a base64 data URI string for API transfer."""
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"
