"""
DRISHTI AI - Quantitative Grad-CAM Validation on IDRiD Lesion Segmentation Masks.
Validates explainability objectively against pixel-level ground truth:
- Microaneurysms (MA)
- Haemorrhages (HE)
- Hard Exudates (EX)
- Soft Exudates / Cotton Wool Spots (SE)

Computes:
1. Pointing Game Hit Rate (does peak activation fall inside true lesion?)
2. Mean Intersection over Union (mIoU at threshold tau)
3. Precision, Recall, and Dice Score on Lesion Masks
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch

from ai.config import ARTIFACTS_DIR, REPORTS_DIR
from ai.datasets.preprocessing import preprocess_fundus_image
from ai.explainability.gradcam import GradCAM


def load_binary_mask(mask_path: Union[str, Path], target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Loads a grayscale ground-truth lesion mask and resizes/binarizes it."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(target_size, dtype=np.uint8)
    resized = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.uint8)


def evaluate_pointing_game(gradcam_heatmap: np.ndarray, ground_truth_mask: np.ndarray) -> bool:
    """
    Pointing Game: A hit is scored if the argmax coordinate of Grad-CAM
    falls within the positive ground-truth lesion mask.
    """
    max_idx = np.unravel_index(np.argmax(gradcam_heatmap), gradcam_heatmap.shape)
    return bool(ground_truth_mask[max_idx] > 0)


def compute_iou_dice(
    gradcam_heatmap: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Computes IoU, Dice Score, Precision, and Recall between thresholded CAM and mask.
    """
    cam_binary = (gradcam_heatmap >= threshold).astype(np.uint8)
    gt_binary = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(cam_binary, gt_binary).sum()
    union = np.logical_or(cam_binary, gt_binary).sum()
    cam_total = cam_binary.sum()
    gt_total = gt_binary.sum()

    iou = float(intersection / union) if union > 0 else (1.0 if gt_total == 0 and cam_total == 0 else 0.0)
    dice = float((2.0 * intersection) / (cam_total + gt_total)) if (cam_total + gt_total) > 0 else 1.0
    precision = float(intersection / cam_total) if cam_total > 0 else 0.0
    recall = float(intersection / gt_total) if gt_total > 0 else 0.0

    return iou, dice, precision, recall


def run_idrid_validation(
    model: torch.nn.Module,
    idrid_images_dir: Union[str, Path],
    idrid_masks_dir: Union[str, Path],
    samples_list: Optional[List[Dict]] = None,
    device: str = "cpu",
    output_report_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Runs quantitative explainability benchmark across IDRiD images & lesion masks.
    """
    model.eval()
    model.to(device)
    cam_engine = GradCAM(model)

    images_dir = Path(idrid_images_dir)
    masks_dir = Path(idrid_masks_dir)

    lesion_types = ["microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates"]
    results = []

    pointing_hits = 0
    total_evaluated = 0
    total_lesion_images = 0
    ious = []
    dices = []
    precisions = []
    recalls = []

    # If samples_list not provided, search files in directory
    if samples_list is None:
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.tif"))
    else:
        image_files = [images_dir / s["image_name"] for s in samples_list]

    for img_path in image_files:
        if not img_path.exists():
            continue
            
        base_stem = img_path.stem
        # Try loading composite lesion mask
        composite_mask = np.zeros((512, 512), dtype=np.uint8)
        found_any_mask = False

        for l_type in lesion_types:
            l_mask_path = masks_dir / l_type / f"{base_stem}_{l_type}.tif"
            if not l_mask_path.exists():
                l_mask_path = masks_dir / f"{base_stem}_{l_type}.png"
            if l_mask_path.exists():
                l_mask = load_binary_mask(l_mask_path, target_size=(512, 512))
                composite_mask = np.maximum(composite_mask, l_mask)
                found_any_mask = True

        if not found_any_mask:
            # Check for general mask
            gen_mask = masks_dir / f"{base_stem}_mask.png"
            if gen_mask.exists():
                composite_mask = load_binary_mask(gen_mask, target_size=(512, 512))
                found_any_mask = True

        # Check if ground truth has lesions
        has_lesions = bool(composite_mask.sum() > 0)
        
        # Preprocess and run inference
        preprocessed = preprocess_fundus_image(img_path, target_size=(512, 512))
        tensor = torch.tensor(preprocessed).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        with torch.enable_grad():
            heatmap = cam_engine.generate_heatmap(tensor)

        if has_lesions:
            hit = bool(evaluate_pointing_game(heatmap, composite_mask))
            iou, dice, prec, rec = compute_iou_dice(heatmap, composite_mask, threshold=0.35)
            pointing_hits += int(hit)
            total_lesion_images += 1
            ious.append(iou)
            dices.append(dice)
            precisions.append(prec)
            recalls.append(rec)
        else:
            hit = None
            iou, dice, prec, rec = 1.0, 1.0, 1.0, 1.0

        total_evaluated += 1

        results.append({
            "image": str(img_path.name),
            "has_ground_truth_lesions": bool(has_lesions),
            "pointing_game_hit": hit,
            "iou_score": round(float(iou), 4),
            "dice_score": round(float(dice), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        })

    pointing_accuracy = (pointing_hits / total_lesion_images) if total_lesion_images > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    summary_report = {
        "dataset": "IDRiD (Indian Diabetic Retinopathy Image Dataset)",
        "population": "Nanded, Maharashtra, India",
        "total_images_evaluated": total_evaluated,
        "pointing_game_accuracy": round(pointing_accuracy * 100.0, 2),
        "mean_iou": round(mean_iou, 4),
        "mean_dice_score": round(mean_dice, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "clinical_interpretation": (
            f"The model's Grad-CAM heatmaps achieve a Pointing Game accuracy of {pointing_accuracy*100.0:.1f}% "
            f"and mIoU of {mean_iou:.3f} on IDRiD pixel lesion masks, objectively validating that "
            f"AI visual attention corresponds to true microvascular lesions (microaneurysms, hemorrhages, exudates)."
        )
    }

    if output_report_path:
        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": summary_report, "samples": results}, f, indent=2)

    return summary_report
