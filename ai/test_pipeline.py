"""
DRISHTI AI - Comprehensive Automated Test & Verification Suite.
Validates:
1. Ben Graham Preprocessing + CLAHE + Circular Fundus Masking
2. EfficientNet-B3 Model Architecture & Parameter Counts (and scalable B4/ConvNeXt)
3. QWK & Weighted Cross-Entropy Loss Computation
4. Grad-CAM Visual Heatmap & Overlay Generation
5. Quantitative Explainability Validation on IDRiD Lesion Ground Truth
6. External Generalization Benchmark on Messidor-2
7. Full Inference Engine (Quality Check -> Preprocessing -> DR Grade -> Risk Triage -> Grad-CAM)
8. TorchScript Model Serialization & Export
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch

from ai.config import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR
from ai.datasets.preprocessing import (
    crop_fundus_circular,
    ben_graham_normalization,
    apply_green_clahe,
    preprocess_fundus_image,
    get_inference_tensor,
)
from ai.explainability.gradcam import GradCAM, overlay_gradcam
from ai.explainability.validate_idrid import run_idrid_validation
from ai.export import export_model_for_serving
from ai.inference import DrishtiInferenceEngine
from ai.models.backbone import build_model
from ai.models.loss import (
    HybridQWKCELoss,
    QuadraticWeightedKappaLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
    compute_qwk_metric,
)
from ai.models.quality_checker import HeuristicQualityChecker
from ai.validation.evaluate_messidor import evaluate_messidor_generalization


def test_preprocessing():
    print("\n[1/8] Testing Clinical Preprocessing Pipeline...")
    sample_img_path = DATA_DIR / "aptos2019" / "train_images" / "aptos_sample_g2_00.png"
    assert sample_img_path.exists(), f"Sample image not found: {sample_img_path}"

    raw_img = cv2.imread(str(sample_img_path))
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    cropped = crop_fundus_circular(raw_rgb)
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0, "Circular crop failed"

    bg_norm = ben_graham_normalization(cropped, sigma=30)
    assert bg_norm.shape == cropped.shape, "Ben Graham normalization shape mismatch"

    clahe_img = apply_green_clahe(bg_norm)
    assert clahe_img.shape == cropped.shape, "CLAHE shape mismatch"

    full_proc = preprocess_fundus_image(sample_img_path, target_size=(512, 512))
    assert full_proc.shape == (512, 512, 3), f"Full preprocess shape incorrect: {full_proc.shape}"

    tensor = get_inference_tensor(sample_img_path, target_size=(512, 512))
    assert tensor.shape == (1, 3, 512, 512), f"Inference tensor shape mismatch: {tensor.shape}"

    print("   ✓ Circular crop, Ben Graham subtraction, CLAHE, and Tensor conversion passed!")


def test_model_architecture():
    print("\n[2/8] Testing Model Architecture & Parameter Scalability...")
    model_b3 = build_model(backbone_name="efficientnet_b3", num_classes=5, pretrained=False)
    params_b3 = model_b3.count_parameters()
    print(f"   ✓ EfficientNet-B3 Total Parameters: {params_b3['total_parameters']:,} ({params_b3['total_parameters']/1e6:.2f}M)")

    model_b4 = build_model(backbone_name="efficientnet_b4", num_classes=5, pretrained=False)
    params_b4 = model_b4.count_parameters()
    print(f"   ✓ EfficientNet-B4 Total Parameters: {params_b4['total_parameters']:,} ({params_b4['total_parameters']/1e6:.2f}M)")

    dummy_tensor = torch.randn(2, 3, 512, 512)
    output_b3 = model_b3(dummy_tensor)
    assert output_b3.shape == (2, 5), f"Model output shape incorrect: {output_b3.shape}"
    print("   ✓ Forward pass tensor shape (2, 5) verified!")


def test_loss_functions():
    print("\n[3/8] Testing Loss Functions & Metric Alignment...")
    logits = torch.tensor([
        [2.5, 0.1, -1.0, -2.0, -3.0],  # Confident Grade 0
        [-2.0, -1.0, 0.5, 3.2, 0.1],   # Confident Grade 3
    ], requires_grad=True)
    targets = torch.tensor([0, 3])

    # QWK Loss
    qwk_loss_fn = QuadraticWeightedKappaLoss(num_classes=5)
    loss_qwk = qwk_loss_fn(logits, targets)
    loss_qwk.backward()
    assert not torch.isnan(loss_qwk).item(), "QWK loss returned NaN"
    print(f"   ✓ Quadratic Weighted Kappa Loss value: {loss_qwk.item():.4f}")

    # Class weights calculation
    dummy_labels = np.array([0, 0, 0, 0, 0, 1, 2, 2, 3, 4])
    weights = compute_class_weights(dummy_labels, num_classes=5)
    assert weights.shape == (5,), "Class weights shape mismatch"
    print(f"   ✓ Inverse frequency class weights: {weights.numpy().round(2)}")

    # QWK metric test
    y_true = np.array([0, 1, 2, 3, 4, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 4, 2, 3])
    qwk_score = compute_qwk_metric(y_true, y_pred)
    assert 0.0 <= qwk_score <= 1.0, f"QWK score out of bounds: {qwk_score}"
    print(f"   ✓ Cohen's QWK metric calculation verified: {qwk_score:.4f}")


def test_gradcam_generation():
    print("\n[4/8] Testing Grad-CAM Visual Explainability...")
    model = build_model(backbone_name="efficientnet_b3", num_classes=5, pretrained=False)
    gradcam = GradCAM(model)

    sample_img_path = DATA_DIR / "aptos2019" / "train_images" / "aptos_sample_g3_00.png"
    tensor = get_inference_tensor(sample_img_path)

    heatmap = gradcam.generate_heatmap(tensor, target_class=3)
    assert heatmap.shape == (512, 512), f"Heatmap shape mismatch: {heatmap.shape}"
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0, "Heatmap values not normalized to [0, 1]"

    proc_rgb = preprocess_fundus_image(sample_img_path)
    overlay = overlay_gradcam(proc_rgb, heatmap, alpha=0.45)
    assert overlay.shape == (512, 512, 3), f"Overlay shape mismatch: {overlay.shape}"

    # Save test overlay
    test_out = ARTIFACTS_DIR / "test_gradcam_overlay.png"
    cv2.imwrite(str(test_out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"   ✓ Grad-CAM heatmap generated and saved to: {test_out}")


def test_idrid_quantitative_validation():
    print("\n[5/8] Testing Quantitative Explainability Validation on IDRiD Ground Truth...")
    model = build_model(backbone_name="efficientnet_b3", num_classes=5, pretrained=False)
    
    idrid_images_dir = DATA_DIR / "idrid" / "images"
    idrid_masks_dir = DATA_DIR / "idrid" / "masks"
    report_out = REPORTS_DIR / "idrid_validation_report.json"

    report = run_idrid_validation(
        model=model,
        idrid_images_dir=idrid_images_dir,
        idrid_masks_dir=idrid_masks_dir,
        output_report_path=report_out,
    )

    assert "pointing_game_accuracy" in report, "Pointing game accuracy missing"
    assert "mean_iou" in report, "Mean IoU missing"
    print(f"   ✓ IDRiD Pointing Game Accuracy: {report['pointing_game_accuracy']}%")
    print(f"   ✓ IDRiD Mean IoU on Lesion Masks: {report['mean_iou']}")
    print(f"   ✓ Summary Report written to: {report_out}")


def test_messidor2_generalization():
    print("\n[6/8] Testing External Generalization Benchmark on Messidor-2...")
    model = build_model(backbone_name="efficientnet_b3", num_classes=5, pretrained=False)
    
    csv_path = DATA_DIR / "messidor2" / "messidor2_grades.csv"
    images_dir = DATA_DIR / "messidor2" / "images"
    report_out = REPORTS_DIR / "messidor2_benchmark_report.json"

    report = evaluate_messidor_generalization(
        model=model,
        csv_path=csv_path,
        images_dir=images_dir,
        output_report_path=report_out,
    )

    assert "quadratic_weighted_kappa" in report, "QWK missing in Messidor-2 report"
    assert "referable_dr_metrics" in report, "Referable DR metrics missing"
    print(f"   ✓ Messidor-2 Generalization QWK: {report['quadratic_weighted_kappa']}")
    print(f"   ✓ Referable DR Sensitivity: {report['referable_dr_metrics']['sensitivity']}%")
    print(f"   ✓ Referable DR Specificity: {report['referable_dr_metrics']['specificity']}%")
    print(f"   ✓ Benchmark Report written to: {report_out}")


def test_model_export():
    print("\n[7/8] Testing Production Model Serialization & TorchScript Export...")
    weights_path = MODELS_DIR / "drishti_test_model.pth"
    model = build_model(backbone_name="efficientnet_b3", num_classes=5, pretrained=False)
    torch.save({"state_dict": model.state_dict()}, weights_path)

    manifest = export_model_for_serving(
        weights_path=weights_path,
        output_dir=MODELS_DIR,
        backbone_name="efficientnet_b3",
    )

    assert Path(manifest["production_checkpoint"]).exists(), "Checkpoint export missing"
    assert Path(manifest["torchscript_traced"]).exists(), "TorchScript export missing"
    print("   ✓ Exported clean state_dict and traced TorchScript .pt successfully!")


def test_full_inference_engine():
    print("\n[8/8] Testing Unified DrishtiInferenceEngine...")
    weights_path = MODELS_DIR / "drishti_test_model.pth"
    engine = DrishtiInferenceEngine(
        weights_path=weights_path,
        backbone_name="efficientnet_b3",
        enable_gradcam=True,
    )

    sample_img_path = DATA_DIR / "aptos2019" / "train_images" / "aptos_sample_g4_00.png"
    result = engine.analyze(sample_img_path)

    assert result["status"] == "SUCCESS", "Inference status not SUCCESS"
    assert "quality_assessment" in result, "Quality assessment missing"
    assert "clinical_prediction" in result, "Clinical prediction missing"
    assert "referral_triage" in result, "Referral triage missing"
    assert "explainability" in result, "Explainability payload missing"
    assert result["explainability"]["has_overlay"], "Grad-CAM overlay missing"

    print(f"   ✓ Image Quality Status: {result['quality_assessment']['status']}")
    print(f"   ✓ Predicted ICDR Grade: {result['clinical_prediction']['icdr_grade']} ({result['clinical_prediction']['icdr_label']})")
    print(f"   ✓ Referral Risk Bucket: {result['referral_triage']['risk_bucket']} (Urgency: {result['referral_triage']['urgency']})")
    print(f"   ✓ Clinical Action: {result['referral_triage']['actionable_guidance']}")
    print(f"   ✓ Grad-CAM Overlay Generated: {result['explainability']['has_overlay']}")


def run_all_tests():
    print("=" * 70)
    print("      DRISHTI AI — PHASE 1 AI/ML VERIFICATION TEST SUITE      ")
    print("=" * 70)
    test_preprocessing()
    test_model_architecture()
    test_loss_functions()
    test_gradcam_generation()
    test_idrid_quantitative_validation()
    test_messidor2_generalization()
    test_model_export()
    test_full_inference_engine()
    print("\n" + "=" * 70)
    print("🎉 ALL PHASE 1 PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
