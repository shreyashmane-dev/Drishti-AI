"""
DRISHTI AI - Clinical Loss Functions for Ordinal Retinopathy Classification.
Implements:
1. Continuous Quadratic Weighted Kappa (QWK) Loss (APTOS/EyePACS primary metric)
2. Weighted Cross-Entropy Loss (combats severe class imbalance where Grade 0 dominates)
3. Hybrid QWK + Cross-Entropy Loss
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score


class QuadraticWeightedKappaLoss(nn.Module):
    """
    Continuous, differentiable approximation of Cohen's Quadratic Weighted Kappa.
    Unlike standard cross-entropy which treats all misclassifications equally,
    QWK penalizes distant mistakes quadratically (e.g. predicting Grade 0 for Grade 4
    is penalized 16x more than predicting Grade 3 for Grade 4).
    """
    def __init__(self, num_classes: int = 5, epsilon: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # Build quadratic penalty weight matrix W_ij = ((i - j)^2) / ((N - 1)^2)
        indices = torch.arange(num_classes).float()
        w = (indices.unsqueeze(0) - indices.unsqueeze(1)) ** 2
        w = w / ((num_classes - 1) ** 2)
        self.register_buffer("weight_matrix", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw model outputs
            targets: (B,) class indices 0..num_classes-1
        """
        probs = F.softmax(logits, dim=1)
        batch_size = logits.size(0)

        # One-hot encode ground truth targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Observed confusion matrix: O = P^T * Y
        observed = torch.matmul(probs.t(), targets_one_hot) / batch_size

        # Expected confusion matrix under independence: E = (P^T * 1) * (1^T * Y)
        hist_pred = probs.sum(dim=0, keepdim=True) / batch_size
        hist_target = targets_one_hot.sum(dim=0, keepdim=True) / batch_size
        expected = torch.matmul(hist_pred.t(), hist_target)

        # Compute weighted sum
        numerator = torch.sum(self.weight_matrix * observed)
        denominator = torch.sum(self.weight_matrix * expected) + self.epsilon

        kappa = 1.0 - (numerator / denominator)
        # Loss is 1 - kappa (minimizing loss maximizes Kappa)
        return 1.0 - kappa


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with class reweighting to compensate for DR dataset imbalances.
    """
    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class HybridQWKCELoss(nn.Module):
    """
    Combined loss function:
    L = alpha * L_QWK + (1 - alpha) * L_WCE
    Provides smooth cross-entropy gradients during early epochs plus QWK metric alignment.
    """
    def __init__(
        self,
        num_classes: int = 5,
        class_weights: torch.Tensor = None,
        alpha: float = 0.5
    ):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.qwk_loss = QuadraticWeightedKappaLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        qwk = self.qwk_loss(logits, targets)
        return self.alpha * qwk + (1.0 - self.alpha) * ce


class DualTaskClinicalLoss(nn.Module):
    """
    Robust Dual-Task Clinical Loss Function for Diabetic Retinopathy.
    Combines:
    1. Label-Smoothed Categorical Cross-Entropy (prevents single-class mode collapse)
    2. Continuous Smooth L1 (Huber) Ordinal Loss (enforces natural disease severity distance)
    L_total = L_ce + lambda_ord * L_ordinal
    """
    def __init__(
        self,
        num_classes: int = 5,
        label_smoothing: float = 0.1,
        ordinal_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        self.ordinal_loss = nn.SmoothL1Loss(beta=0.5)

    def forward(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        targets: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(outputs, tuple):
            logits, ordinal_pred = outputs
            ce = self.ce_loss(logits, targets)
            ord_loss = self.ordinal_loss(ordinal_pred.view(-1), targets.float().view(-1))
            return ce + self.ordinal_weight * ord_loss
        else:
            return self.ce_loss(outputs, targets)


def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """
    Computes inverse frequency class weights:
    w_c = total_samples / (num_classes * count_c)
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    # Normalize weights so mean is 1.0
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def compute_qwk_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes exact scikit-learn Quadratic Weighted Kappa between integer predictions.
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
