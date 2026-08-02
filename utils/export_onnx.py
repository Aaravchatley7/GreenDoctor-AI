"""ONNX Export & INT8 Quantization Utility for Explainable AI Plant Disease Detection.

Converts trained EfficientNet-B0 model PyTorch weights (best_model.pth) to ONNX format
and applies INT8 dynamic quantization for efficient edge and mobile deployment.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # type: ignore

from config.config import TrainingConfig
from models.efficientnet import build_efficientnet_b0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("utils.export_onnx")


def export_model_to_onnx(config: TrainingConfig) -> Path:
    """Exports trained PyTorch EfficientNet-B0 model to ONNX format.

    Args:
        config: TrainingConfig instance.

    Returns:
        Path: Path to exported ONNX model file.
    """
    checkpoint_path = config.MODELS_DIR / config.BEST_MODEL_NAME
    class_names_path = config.MODELS_DIR / config.CLASS_NAMES_FILE

    if not checkpoint_path.exists():
        logger.error("Model checkpoint not found at %s. Run training first.", checkpoint_path)
        sys.exit(1)

    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
            num_classes = len(class_names)
    else:
        num_classes = 15

    # 1. Instantiate and load PyTorch model
    logger.info("Loading PyTorch checkpoint from %s...", checkpoint_path)
    device = torch.device("cpu")
    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Prepare dummy input tensor
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], device=device)
    onnx_output_path = config.MODELS_DIR / "best_model.onnx"

    # 3. Export ONNX graph
    logger.info("Exporting model to ONNX at %s...", onnx_output_path)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    onnx_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    logger.info("Successfully exported ONNX model! File size: %.2f MB", onnx_size_mb)

    # 4. Apply PyTorch INT8 Dynamic Quantization
    try:
        logger.info("Applying PyTorch INT8 Dynamic Quantization...")
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized_path = config.MODELS_DIR / "best_model_quantized.pth"
        torch.save(quantized_model.state_dict(), quantized_path)
        quantized_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
        logger.info("Successfully created INT8 Quantized model! File size: %.2f MB", quantized_size_mb)
    except Exception as exc:
        logger.warning("INT8 Quantization notice: %s", exc)

    return onnx_output_path


if __name__ == "__main__":
    cfg = TrainingConfig()
    export_model_to_onnx(cfg)
