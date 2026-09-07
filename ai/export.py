"""
DRISHTI AI - Model Serialization & Production Export Module.
Exports trained weights to TorchScript and ONNX formats for FastAPI backend serving.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ai.config import MODELS_DIR
from ai.models.backbone import build_model


def export_model_for_serving(
    weights_path: Union[str, Path],
    output_dir: Union[str, Path] = MODELS_DIR,
    backbone_name: str = "efficientnet_b3",
    num_classes: int = 5,
    input_shape: tuple = (1, 3, 512, 512),
) -> Dict[str, str]:
    """
    Exports a trained PyTorch model to:
    1. Clean inference checkpoint (state_dict + config metadata)
    2. Traced TorchScript (.pt)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(weights_path)

    # Initialize architecture
    model = build_model(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=False,
        device="cpu",
    )

    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from: {weights_path}")
    else:
        print(f"Weights file not found at {weights_path}, exporting architecture with initial weights.")

    model.eval()

    # 1. Clean production state_dict
    export_pth = output_dir / f"drishti_retinopathy_{backbone_name}.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "backbone_name": backbone_name,
        "num_classes": num_classes,
        "input_resolution": input_shape[2:],
        "version": "1.0.0",
    }, export_pth)

    # 2. TorchScript Traced Export
    dummy_input = torch.randn(*input_shape)
    traced_model = torch.jit.trace(model, dummy_input)
    export_torchscript = output_dir / f"drishti_retinopathy_{backbone_name}_traced.pt"
    traced_model.save(str(export_torchscript))

    manifest = {
        "production_checkpoint": str(export_pth),
        "torchscript_traced": str(export_torchscript),
        "backbone": backbone_name,
        "num_classes": num_classes,
        "input_resolution": list(input_shape[2:]),
    }

    manifest_path = output_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Successfully exported models:")
    print(f"   Checkpoint: {export_pth}")
    print(f"   TorchScript: {export_torchscript}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DRISHTI AI Model for Serving")
    parser.add_argument("--weights", type=str, default=str(MODELS_DIR / "drishti_best_model.pth"))
    parser.add_argument("--backbone", type=str, default="efficientnet_b3")
    args = parser.parse_args()
    export_model_for_serving(weights_path=args.weights, backbone_name=args.backbone)
