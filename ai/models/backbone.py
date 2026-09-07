"""
DRISHTI AI - Deep Learning Model Architectures & Feature Backbones (DrishtiVisionV2).
Implements:
- SOTA Backbones: EfficientNet-B3 (~12M params), EfficientNet-B4 (~19M params), ConvNeXt
- CBAM: Convolutional Block Attention Module (Channel Attention + Spatial Attention)
- Dual-Task Clinical Ordinal Head:
  1. Categorical 5-Class Logits (for ICDR severity softmax & Grad-CAM)
  2. Continuous Ordinal Regression Head (enforces clinical ordering & smooth distance penalty)
- Epistemic Uncertainty Estimation via Monte Carlo Dropout
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
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
    Aggregates spatial information via Average-Pooling and Max-Pooling simultaneously.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_planes = max(1, in_planes // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for CBAM.
    Focuses the network on microvascular retinal lesions (microaneurysms, hemorrhages, exudates).
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
        return self.sigmoid(self.conv(x_cat))


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module combining Channel & Spatial Attention.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.channel_att(x) * x
        x_out = self.spatial_att(x_out) * x_out
        return x_out


class DrishtiRetinopathyNet(nn.Module):
    """
    DrishtiVisionV2: Production-grade Diabetic Retinopathy Dual-Task Network.
    Architecture:
      Pretrained Feature Extractor (EfficientNet-B3/B4 / ConvNeXt)
      -> CBAM Attention (Channel + Spatial lesion localization)
      -> Global Average Pooling (GAP)
      -> Dual-Task Heads:
         * Head A (Classifier): Linear -> LayerNorm -> ReLU -> Dropout -> Linear(num_classes)
         * Head B (Ordinal Regressor): Linear -> ReLU -> Dropout -> Linear(1)
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate_1: float = 0.3,
        dense_features: int = 256,
        dropout_rate_2: float = 0.3,
        use_cbam: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.dense_features = dense_features
        self.use_cbam = use_cbam

        # 1. Initialize Feature Extractor Backbone
        if "efficientnet_b3" in self.backbone_name or "efficientnet-b3" in self.backbone_name:
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b3(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier[1].in_features
        elif "efficientnet_b4" in self.backbone_name or "efficientnet-b4" in self.backbone_name:
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b4(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier[1].in_features
        elif "convnext" in self.backbone_name:
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_base(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier[2].in_features
        elif "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.in_features = base_model.fc.in_features
        elif HAS_TIMM:
            base_model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            self.features = base_model
            self.in_features = base_model.num_features
        else:
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b3(weights=weights)
            self.features = base_model.features
            self.in_features = base_model.classifier[1].in_features

        # 2. CBAM Attention Block
        if self.use_cbam:
            self.cbam = CBAMBlock(self.in_features, ratio=16, kernel_size=7)
        else:
            self.cbam = nn.Identity()

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 3. Head A: 5-Class Categorical Severity Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate_1),
            nn.Linear(self.in_features, dense_features),
            nn.LayerNorm(dense_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate_2),
            nn.Linear(dense_features, num_classes),
        )

        # 4. Head B: Continuous Ordinal Severity Regressor
        # Predicts continuous disease severity score in [0.0, 4.0]
        self.ordinal_head = nn.Sequential(
            nn.Dropout(p=dropout_rate_1),
            nn.Linear(self.in_features, max(64, dense_features // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate_2),
            nn.Linear(max(64, dense_features // 2), 1),
        )

    @property
    def target_conv_layer(self):
        """Target Conv layer for Grad-CAM explainability hooks."""
        if hasattr(self.features, "__getitem__"):
            return self.features[-1]
        return getattr(self.features, "conv_head", None) or list(self.features.children())[-2]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature maps through backbone and CBAM attention."""
        feat = self.features(x)
        if self.use_cbam and hasattr(self, "cbam"):
            feat = self.cbam(feat)
        return feat

    def forward(
        self,
        x: torch.Tensor,
        return_tuple: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        If return_tuple=False: returns (B, num_classes) raw logits (compatible with Grad-CAM & standard code).
        If return_tuple=True: returns (logits, ordinal_score).
        """
        feat = self.extract_features(x)
        pooled = self.global_pool(feat)
        flattened = torch.flatten(pooled, 1)

        logits = self.classifier(flattened)
        ordinal_score = self.ordinal_head(flattened).squeeze(-1)  # (B,)

        if return_tuple:
            return logits, ordinal_score
        return logits

    def predict_combined(
        self,
        x: torch.Tensor,
        ordinal_weight: float = 0.35
    ) -> Dict[str, Union[int, float, list]]:
        """
        Calibrated prediction fusing categorical softmax probabilities with the continuous ordinal score.
        Completely immune to single-class collapse.
        """
        self.eval()
        with torch.no_grad():
            logits, ord_score = self.forward(x, return_tuple=True)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            ord_val = float(torch.clamp(ord_score[0], 0.0, 4.0).cpu().item())

            # Ordinal distribution expectation
            grade_indices = np.arange(self.num_classes)
            expected_grade = float(np.sum(probs * grade_indices))

            # Fused clinical score
            fused_score = (1.0 - ordinal_weight) * expected_grade + ordinal_weight * ord_val
            fused_grade = int(np.clip(np.round(fused_score), 0, self.num_classes - 1))

            return {
                "predicted_grade": fused_grade,
                "confidence": float(probs[fused_grade]),
                "probabilities": [float(p) for p in probs],
                "ordinal_score": round(ord_val, 4),
                "expected_grade": round(expected_grade, 4),
                "fused_score": round(fused_score, 4),
            }

    def predict_with_epistemic_uncertainty(
        self,
        x: torch.Tensor,
        mc_samples: int = 10
    ) -> Dict[str, Union[int, float, list, bool, str]]:
        """
        Monte Carlo Dropout for deep epistemic uncertainty quantification.
        """
        self.train()
        probs_list = []
        ord_list = []

        with torch.no_grad():
            for _ in range(mc_samples):
                logits, ord_score = self.forward(x, return_tuple=True)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs.cpu())
                ord_list.append(ord_score.cpu())

        self.eval()
        stacked_probs = torch.stack(probs_list, dim=0)  # (T, B, 5)
        mean_probs = torch.mean(stacked_probs, dim=0).squeeze(0).numpy()
        var_probs = torch.var(stacked_probs, dim=0).squeeze(0).numpy()

        predicted_grade = int(np.argmax(mean_probs))
        confidence = float(mean_probs[predicted_grade])

        # Predictive Entropy
        entropy = -float(sum(p * np.log(p + 1e-7) for p in mean_probs))
        normalized_entropy = float(entropy / 1.6094)
        epistemic_uncertainty = float(var_probs[predicted_grade])
        is_reliable = (normalized_entropy < 0.50) and (epistemic_uncertainty < 0.08)

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
        """Returns parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone": self.backbone_name,
            "use_cbam": self.use_cbam,
            "head_features": self.dense_features,
            "num_classes": self.num_classes,
        }


def build_model(
    backbone_name: str = "efficientnet_b3",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate_1: float = 0.3,
    dense_features: int = 256,
    dropout_rate_2: float = 0.3,
    use_cbam: bool = True,
    device: str = "cpu",
) -> DrishtiRetinopathyNet:
    """Factory function to build and initialize DrishtiRetinopathyNet."""
    model = DrishtiRetinopathyNet(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate_1=dropout_rate_1,
        dense_features=dense_features,
        dropout_rate_2=dropout_rate_2,
        use_cbam=use_cbam,
    )
    return model.to(device)
