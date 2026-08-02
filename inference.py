"""CLI Inference & Testing Script for Explainable AI Plant Disease Detection.

Run inference on any single leaf image file to get disease diagnosis, confidence score,
Grad-CAM visual heatmap overlay, and Groq LLM clinical advice.

Usage:
    .venv/bin/python inference.py --image path/to/leaf_image.jpg
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.llm_explainer import get_disease_explanation
from config.config import TrainingConfig
from explainability.grad_cam import GradCAM
from explainability.visualizer import overlay_heatmap_on_image
from models.efficientnet import build_efficientnet_b0
from preprocessing.transforms import preprocess_image_for_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference")


def predict_single_image(
    image_path: Path,
    config: TrainingConfig,
    output_heatmap_path: Path | None = None,
) -> None:
    """Performs inference and Grad-CAM generation on a single leaf image.

    Args:
        image_path: Path to input leaf image file.
        config: Central TrainingConfig instance.
        output_heatmap_path: Optional output path to save Grad-CAM overlay image.
    """
    if not image_path.exists():
        logger.error("Specified image file '%s' does not exist.", image_path)
        sys.exit(1)

    class_names_path = config.MODELS_DIR / config.CLASS_NAMES_FILE
    model_checkpoint_path = config.MODELS_DIR / config.BEST_MODEL_NAME

    if not class_names_path.exists():
        logger.error("Class names file not found at '%s'. Train the model first.", class_names_path)
        sys.exit(1)

    if not model_checkpoint_path.exists():
        logger.error("Model checkpoint not found at '%s'. Train the model first.", model_checkpoint_path)
        sys.exit(1)

    # 1. Load Class Names
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    # 2. Select Device & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Running inference on device: %s", device)

    model = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load & Preprocess Image
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
    
    input_tensor = preprocess_image_for_inference(img_rgb).to(device)

    # 4. Perform Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf_val, pred_idx_tensor = torch.max(probs, dim=1)
        pred_idx = int(pred_idx_tensor.item())
        confidence = float(conf_val.item()) * 100.0

    predicted_class = class_names[pred_idx]

    # 5. Generate Grad-CAM Heatmap
    target_layer = model.get_target_layer_for_gradcam()
    grad_cam = GradCAM(model=model, target_layer=target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor=input_tensor, target_class_idx=pred_idx)
    overlay_img = overlay_heatmap_on_image(original_image=img_rgb, heatmap=heatmap, alpha=0.5)

    if output_heatmap_path is None:
        output_heatmap_path = PROJECT_ROOT / "reports" / f"gradcam_{image_path.stem}.png"
    
    output_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_img.save(output_heatmap_path)

    # 6. Fetch Groq LLM Insights
    insights = get_disease_explanation(class_name=predicted_class, confidence=confidence)

    # 7. Display Clean Terminal Output
    print("\n" + "=" * 65)
    print("      EXPLAINABLE AI (XAI) PLANT DISEASE DIAGNOSIS REPORT        ")
    print("=" * 65)
    print(f" 📄 Image File     : {image_path.name}")
    print(f" 🌿 Diagnosed Class: {predicted_class}")
    print(f" 🩺 Clean Title    : {insights['disease_title']}")
    print(f" 🎯 Confidence     : {confidence:.2f}%")
    print(f" 👁️ Grad-CAM Overlay: Saved to {output_heatmap_path}")
    print("-" * 65)
    print(" 🧪 Scientific Explanation:")
    print(f"    {insights['explanation']}")
    print("-" * 65)
    print(" 💊 Treatment & Cure:")
    print(f"    {insights['cure']}")
    print("-" * 65)
    print(" 🛡️ Long-term Prevention:")
    print(f"    {insights['prevention']}")
    print("-" * 65)
    print(" ⚠️ Immediate Precautions:")
    print(f"    {insights['precautions']}")
    print("=" * 65 + "\n")


def main() -> None:
    """CLI parser entry point."""
    parser = argparse.ArgumentParser(description="Test trained EfficientNet-B0 model on a single leaf image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image file")
    parser.add_argument("--output-gradcam", type=str, default=None, help="Optional output path for Grad-CAM overlay")
    args = parser.parse_args()

    config = TrainingConfig()
    predict_single_image(
        image_path=Path(args.image),
        config=config,
        output_heatmap_path=Path(args.output_gradcam) if args.output_gradcam else None,
    )


if __name__ == "__main__":
    main()
