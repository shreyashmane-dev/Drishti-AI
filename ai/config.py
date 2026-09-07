"""
DRISHTI AI - System Configuration and Hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Create required directories
for directory in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Clinical Label Scheme (ICDR 5-class Scale)
ICDR_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

# 3-Tier Referral Triage Mapping (Clinical Recommendation Engine)
RISK_BUCKETS = {
    0: {
        "bucket": "Lower-risk",
        "color": "green",
        "action": "Routine annual screening recommended. Maintain glycemic control.",
        "urgency": "Low",
        "referral_needed": False,
    },
    1: {
        "bucket": "Lower-risk",
        "color": "green",
        "action": "Routine follow-up in 6–12 months. Diabetic lifestyle & HbA1c review.",
        "urgency": "Low",
        "referral_needed": False,
    },
    2: {
        "bucket": "Possible DR",
        "color": "amber",
        "action": "Specialist ophthalmologist review recommended within 4–6 weeks.",
        "urgency": "Medium",
        "referral_needed": True,
    },
    3: {
        "bucket": "Higher-risk",
        "color": "red",
        "action": "Priority ophthalmologist referral within 1–2 weeks for laser/anti-VEGF evaluation.",
        "urgency": "High",
        "referral_needed": True,
    },
    4: {
        "bucket": "Higher-risk",
        "color": "red",
        "action": "Urgent specialist referral immediately (<48 hours) to prevent imminent vision loss.",
        "urgency": "Critical",
        "referral_needed": True,
    },
}


@dataclass
class PreprocessingConfig:
    """Fundus image preprocessing pipeline parameters."""
    image_size: Tuple[int, int] = (512, 512)
    crop_tolerance: int = 10  # Circular masking pixel cutoff
    gaussian_kernel_sigma: int = 30  # Ben Graham background subtraction sigma
    gaussian_weight_alpha: float = 4.0
    gaussian_weight_beta: float = -4.0
    gaussian_weight_gamma: float = 128.0
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model architecture & capacity configuration."""
    backbone_name: str = "efficientnet_b3"  # or 'efficientnet_b4', 'convnext_small'
    pretrained: bool = True
    num_classes: int = 5
    dropout_rate_1: float = 0.3
    dense_features: int = 256
    dropout_rate_2: float = 0.3
    target_layer_name: str = "conv_head"  # Layer hooked for Grad-CAM in EfficientNet


@dataclass
class TrainingConfig:
    """Model training hyperparameters and optimization settings."""
    seed: int = 42
    n_splits: int = 5
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    total_epochs: int = 25
    early_stopping_patience: int = 7
    loss_type: str = "weighted_ce"  # 'weighted_ce', 'qwk_loss', or 'focal'
    qwk_loss_alpha: float = 0.5  # Weight balance when combining CE with QWK loss
    gradient_accumulation_steps: int = 1
    device: str = "cuda"  # Auto-falls back to 'cpu' if unavailable


@dataclass
class SystemConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


config = SystemConfig()
