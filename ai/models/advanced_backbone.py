"""
DRISHTI AI - Advanced Multi-Task Dual-Attention Deep Learning Vision System.
Parameters: ~88.6 Million (ConvNeXt-Base) / Scalable to 198M (ConvNeXt-Large).

Core Deep Learning Components:
1. SOTA Vision Backbone (ConvNeXt-Base / Swin Transformer / EfficientNet-B5)
2. CBAM (Convolutional Block Attention Module):
   - Channel Attention (Global Avg & Max Pooling + Shared MLP)
   - Spatial Attention (7x7 Cross-channel convolution for micro-lesion localization)
3. Multi-Task Learning (MTL) Dual Heads:
   - Task A: 5-Class ICDR Diabetic Retinopathy Severity Grading
   - Task B: Binary Referable Diabetic Retinopathy Risk Triage (Multi-task regularization)
4. Deep Epistemic Uncertainty Quantification via Monte Carlo Dropout (MC-Dropout)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for CBAM.
    Computes inter-channel dependencies by aggregating spatial information
    via Average-Pooling and Max-Pooling simultaneously.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for CBAM.
    Focuses the network on 'where' the microvascular lesions (microaneurysms,
    hemorrhages, exudates) are located across the 2D fundus plane.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module combining Channel & Spatial Attention.
    Refines intermediate feature representations dynamically.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Refine across channels
        x_out = self.channel_att(x) * x
        # 2. Refine across spatial dimensions
        x_out = self.spatial_att(x_out) * x_out
        return x_out


class DrishtiAdvancedVisionNet(nn.Module):
    """
    High-Capacity Deep Learning AI Vision Architecture for Retinal Screening.
    - Backbone: ConvNeXt-Base (88.6M parameters) / EfficientNet-B5 (30M parameters)
    - Attention: CBAM (Channel & Spatial Self-Attention)
    - Multi-Task Heads: 5-Class Severity Head + Auxiliary Referable Risk Head
    - Epistemic Uncertainty Estimation via Monte Carlo Dropout
    """
    def __init__(
        self,
        backbone_name: str = "convnext_base",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.35,
        dense_features: int = 512,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Initialize High-Capacity Deep Vision Backbone
        if "convnext_base" in self.backbone_name:
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_base(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier[2].in_features  # 1024 channels
            self.target_conv_layer = self.features[-1]  # Final Conv stage for Grad-CAM
        elif "convnext_large" in self.backbone_name:
            weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_large(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier[2].in_features  # 1536 channels
            self.target_conv_layer = self.features[-1]
        elif "densenet201" in self.backbone_name:
            weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
            base_model = models.densenet201(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier.in_features  # 1920 channels
            self.target_conv_layer = self.features[-1]
        elif "efficientnet_b5" in self.backbone_name:
            weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b5(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier[1].in_features  # 2048 channels
            self.target_conv_layer = self.features[-1]
        elif HAS_TIMM:
            base_model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            self.features = base_model
            self.in_features = base_model.num_features
            self.target_conv_layer = getattr(base_model, "conv_head", None) or list(base_model.children())[-2]
        else:
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_base(weights=weights)
            self.features = base_model.features
            self.in_features = 1024
            self.target_conv_layer = self.features[-1]

        # Convolutional Block Attention Module (CBAM)
        self.cbam = CBAMBlock(self.in_features, ratio=16, kernel_size=7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Multi-Task Head A: 5-Class ICDR Severity Classifier
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.in_features, dense_features),
            nn.LayerNorm(dense_features),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_features, num_classes),
        )

        # Multi-Task Head B: Auxiliary Binary Referable DR Head (Grade >= 2)
        # Joint multi-task training forces the backbone to preserve vision-critical features
        self.auxiliary_referable_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.in_features, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # 0: Non-referable, 1: Referable
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature maps and passes through CBAM Attention."""
        raw_features = self.features(x)
        attended_features = self.cbam(raw_features)
        return attended_features

    def forward(
        self,
        x: torch.Tensor,
        return_auxiliary: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through backbone, CBAM attention, and multi-task heads.
        Returns:
            logits_5class: (B, 5) raw severity logits
            logits_referable: (B, 2) auxiliary referable risk logits (if return_auxiliary=True)
        """
        feat = self.extract_features(x)
        pooled = self.global_pool(feat)
        flattened = torch.flatten(pooled, 1)

        logits_5class = self.classifier_head(flattened)
        logits_referable = self.auxiliary_referable_head(flattened) if return_auxiliary else None

        return logits_5class if not return_auxiliary else (logits_5class, logits_referable)

    def predict_with_epistemic_uncertainty(
        self,
        x: torch.Tensor,
        mc_samples: int = 15
    ) -> Dict[str, Union[int, float, Dict[str, float]]]:
        """
        Monte Carlo (MC) Dropout for Deep Epistemic Uncertainty Estimation.
        Performs T stochastic forward passes with active dropout at inference.
        Returns:
            - Mean prediction probabilities
            - Epistemic Variance (model uncertainty per class)
            - Normalized Predictive Entropy (overall decision ambiguity)
            - Safe Clinical Reliability Flag
        """
        self.train()  # Enable dropout layers for stochastic sampling

        probs_list = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(x, return_auxiliary=False)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs.cpu())

        self.eval()  # Return to evaluation mode

        # Stack into (T, B, num_classes)
        stacked_probs = torch.stack(probs_list, dim=0)  # (T, 1, 5)
        mean_probs = torch.mean(stacked_probs, dim=0).squeeze(0).numpy()
        var_probs = torch.var(stacked_probs, dim=0).squeeze(0).numpy()

        predicted_grade = int(mean_probs.argmax())
        confidence = float(mean_probs[predicted_grade])

        # Compute Shannon Entropy: H = -sum(p * log(p + 1e-7))
        entropy = -float(sum(p * torch.log(torch.tensor(p + 1e-7)) for p in mean_probs))
        # Max entropy for 5 uniform classes is log(5) ≈ 1.609
        normalized_entropy = float(entropy / 1.6094)
        epistemic_uncertainty = float(var_probs[predicted_grade])

        # Flag high-uncertainty cases for human specialist confirmation
        is_reliable = (normalized_entropy < 0.45) and (epistemic_uncertainty < 0.05)

        return {
            "predicted_grade": predicted_grade,
            "confidence": round(confidence, 4),
            "mean_probabilities": [round(float(p), 4) for p in mean_probs],
            "class_variance": [round(float(v), 6) for v in var_probs],
            "epistemic_uncertainty": round(epistemic_uncertainty, 6),
            "predictive_entropy": round(normalized_entropy, 4),
            "is_clinically_reliable": is_reliable,
            "reliability_status": "HIGH_CONFIDENCE" if is_reliable else "BORDERLINE_UNCERTAIN_REFER_TO_DOCTOR",
        }

    def count_parameters(self) -> Dict[str, Union[int, str]]:
        """Returns deep network parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.features.parameters())
        attention_params = sum(p.numel() for p in self.cbam.parameters())
        head_params = sum(p.numel() for p in self.classifier_head.parameters()) + sum(p.numel() for p in self.auxiliary_referable_head.parameters())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": backbone_params,
            "attention_parameters": attention_params,
            "multi_task_head_parameters": head_params,
            "backbone_name": self.backbone_name,
            "million_parameters": f"{total_params / 1e6:.2f}M",
        }


def build_advanced_model(
    backbone_name: str = "convnext_base",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.35,
    dense_features: int = 512,
    device: str = "cpu",
) -> DrishtiAdvancedVisionNet:
    """Factory to build DrishtiAdvancedVisionNet."""
    model = DrishtiAdvancedVisionNet(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        dense_features=dense_features,
    )
    return model.to(device)
