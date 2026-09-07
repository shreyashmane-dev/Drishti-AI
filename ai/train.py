"""
DRISHTI AI - 5-Fold Stratified Cross-Validation Training Pipeline.
Optimizes Quadratic Weighted Kappa (QWK) on APTOS 2019 / EyePACS datasets.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ai.config import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, config
from ai.datasets.augmentations import get_training_transforms, get_validation_transforms
from ai.datasets.unified_dataset import UnifiedFundusDataset, load_aptos_dataframe, load_eyepacs_dataframe
from ai.models.backbone import build_model
from ai.models.advanced_backbone import build_advanced_model
from ai.models.loss import (
    DualTaskClinicalLoss,
    HybridQWKCELoss,
    QuadraticWeightedKappaLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
    compute_qwk_metric,
)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Trains model for one epoch with optional AMP mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    is_dual = isinstance(criterion, DualTaskClinicalLoss)

    for images, targets, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if scaler is not None and str(device).startswith("cuda"):
            with torch.cuda.amp.autocast():
                outputs = model(images, return_tuple=True) if is_dual else model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, return_tuple=True) if is_dual else model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_qwk = compute_qwk_metric(np.array(all_targets), np.array(all_preds))
    return epoch_loss, epoch_qwk


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates model on validation fold."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    is_dual = isinstance(criterion, DualTaskClinicalLoss)

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images, return_tuple=True) if is_dual else model(images)
            loss = criterion(outputs, targets)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_qwk = compute_qwk_metric(np.array(all_targets), np.array(all_preds))
    return val_loss, val_qwk, np.array(all_targets), np.array(all_preds)


def run_training(
    df: pd.DataFrame,
    images_dir: Union[str, Path],
    output_dir: Union[str, Path] = MODELS_DIR,
    backbone_name: str = "efficientnet_b3",
    num_classes: int = 5,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    n_splits: int = 5,
    target_fold: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_amp: bool = True,
    device: Optional[str] = None,
) -> Dict:
    """
    Executes Stratified Cross-Validation training loop on cloud or local hardware.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if num_workers is None:
        num_workers = 2 if (str(device).startswith("cuda") and os.name != "nt") else 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and str(device).startswith("cuda")) else None

    print(f"🚀 Starting DRISHTI AI Training Pipeline")
    print(f"   Backbone: {backbone_name}")
    print(f"   Device: {device} (AMP: {scaler is not None})")
    print(f"   Total samples: {len(df)}")
    print(f"   Workers: {num_workers}, Batch size: {batch_size}")
    if target_fold:
        print(f"   Target Fold: {target_fold} (out of {n_splits}), Epochs: {epochs}")
    else:
        print(f"   Folds: {n_splits}, Epochs per fold: {epochs}")

    train_transforms = get_training_transforms()
    val_transforms = get_validation_transforms()

    # Initialize robust Dual-Task Clinical Loss (CE + Ordinal Smooth L1)
    criterion = DualTaskClinicalLoss(num_classes=num_classes, label_smoothing=0.1, ordinal_weight=0.5).to(device)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    best_overall_qwk = -1.0
    best_model_path = output_dir / "drishti_best_model.pth"

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["diagnosis"])):
        current_fold = fold + 1
        if target_fold is not None and current_fold != target_fold:
            continue

        print(f"\n--- Fold {current_fold}/{n_splits} ---")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = UnifiedFundusDataset(
            df=train_df,
            images_dir=images_dir,
            transform=train_transforms,
        )
        val_dataset = UnifiedFundusDataset(
            df=val_df,
            images_dir=images_dir,
            transform=val_transforms,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
        )

        # Build model
        if "convnext" in backbone_name or "swin" in backbone_name:
            model = build_advanced_model(
                backbone_name=backbone_name,
                num_classes=num_classes,
                pretrained=True,
                device=device,
            )
        else:
            model = build_model(
                backbone_name=backbone_name,
                num_classes=num_classes,
                pretrained=True,
                device=device,
            )

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_fold_qwk = -1.0
        fold_checkpoint = output_dir / f"drishti_{backbone_name}_fold{current_fold}.pth"

        for epoch in range(1, epochs + 1):
            train_loss, train_qwk = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_qwk, _, _ = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} QWK: {train_qwk:.4f} | "
                f"Val Loss: {val_loss:.4f} Val QWK: {val_qwk:.4f}"
            )

            if val_qwk > best_fold_qwk:
                best_fold_qwk = val_qwk
                torch.save({
                    "epoch": epoch,
                    "fold": current_fold,
                    "state_dict": model.state_dict(),
                    "val_qwk": val_qwk,
                    "backbone": backbone_name,
                    "num_classes": num_classes,
                }, fold_checkpoint)

                if val_qwk > best_overall_qwk:
                    best_overall_qwk = val_qwk
                    torch.save({
                        "epoch": epoch,
                        "fold": current_fold,
                        "state_dict": model.state_dict(),
                        "val_qwk": val_qwk,
                        "backbone": backbone_name,
                        "num_classes": num_classes,
                    }, best_model_path)

        fold_results.append({
            "fold": current_fold,
            "best_val_qwk": round(best_fold_qwk, 4),
            "checkpoint": str(fold_checkpoint),
        })

    mean_qwk = np.mean([f["best_val_qwk"] for f in fold_results])
    std_qwk = np.std([f["best_val_qwk"] for f in fold_results])

    summary = {
        "model_architecture": backbone_name,
        "n_folds": n_splits,
        "mean_cv_qwk": round(float(mean_qwk), 4),
        "std_cv_qwk": round(float(std_qwk), 4),
        "best_overall_qwk": round(float(best_overall_qwk), 4),
        "best_model_path": str(best_model_path),
        "folds": fold_results,
    }

    report_path = REPORTS_DIR / "training_cv_summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Training Complete!")
    print(f"   5-Fold Cross-Validation QWK: {mean_qwk:.4f} ± {std_qwk:.4f}")
    print(f"   Best Model Saved to: {best_model_path}")
    return summary


if __name__ == "__main__":
    from ai.config import DATA_DIR
    parser = argparse.ArgumentParser(description="DRISHTI AI 5-Fold Stratified Training")
    parser.add_argument("--csv", type=str, default=None, help="Path to annotations CSV (e.g. real_train.csv)")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory containing fundus images")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing datasets or processed data")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3", help="Model backbone (efficientnet_b3, convnext_base, resnet50)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--target_fold", type=int, default=None, help="Target fold index to train (e.g. 1). Omit to train all folds.")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader num_workers")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR), help="Output directory for checkpoints")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    csv_path = args.csv
    images_dir = args.images_dir

    if csv_path is None or images_dir is None:
        candidates = []
        if args.data_dir:
            dd = Path(args.data_dir)
            candidates.extend([
                (dd / "real_train.csv", dd / "real_images"),
                (dd / "train.csv", dd / "train_images"),
                (dd / "train.csv", dd),
            ])
        candidates.extend([
            (DATA_DIR / "aptos2019" / "real_train.csv", DATA_DIR / "aptos2019" / "real_images"),
            (DATA_DIR / "aptos2019" / "train.csv", DATA_DIR / "aptos2019" / "train_images"),
        ])

        for cand_csv, cand_img in candidates:
            if cand_csv.exists() and cand_img.exists():
                csv_path = str(cand_csv)
                images_dir = str(cand_img)
                break

    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError(f"Could not locate dataset CSV. Please specify --csv or --data_dir.")
    if not images_dir or not Path(images_dir).exists():
        raise FileNotFoundError(f"Could not locate dataset images. Please specify --images_dir or --data_dir.")

    print(f"📊 Using Dataset:\n   CSV: {csv_path}\n   Images: {images_dir}")
    df = pd.read_csv(csv_path)
    if "id_code" in df.columns:
        df = df.rename(columns={"id_code": "image_id"})

    run_training(
        df=df,
        images_dir=images_dir,
        output_dir=args.output_dir,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_splits=args.n_splits,
        target_fold=args.target_fold,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )
