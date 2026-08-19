"""CLI Inference & XAI Visualization Script for PhytoShield AI Plant Disease Detection.

Run inference on any single leaf image to get a disease diagnosis, confidence score,
and heatmap overlays from up to 6 gradient-based XAI methods.

Supported XAI methods:
    gradcam            — Grad-CAM coarse region heatmap                (<5ms)
    gradcam_pp         — Grad-CAM++ second-order gradient heatmap      (<8ms)
    saliency           — Vanilla Saliency pixel-level map              (<3ms)
    smoothgrad         — SmoothGrad denoised saliency (25 samples)     (~50ms)
    integrated_gradients — Integrated Gradients (20-step path)        (~200ms)
    shap               — SHAP 8×8 occlusion patch map                 (~300ms)
    all                — Runs all 6 methods above (LIME excluded from CLI)

Usage:
    .venv/bin/python inference.py --image path/to/leaf.jpg
    .venv/bin/python inference.py --image path/to/leaf.jpg --xai-method all
    .venv/bin/python inference.py --image path/to/leaf.jpg --xai-method shap
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
from explainability.grad_cam_plus_plus import GradCAMPlusPlus
from explainability.integrated_gradients import IntegratedGradients
from explainability.saliency import SaliencyExplainer, SmoothGradExplainer
from explainability.shap_explainer import SHAPExplainer
from explainability.visualizer import overlay_heatmap_on_image
from models.efficientnet import build_efficientnet_b0
from preprocessing.transforms import preprocess_image_for_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference")

# Methods supported in CLI (LIME excluded — requires explicit API call for full control)
_CLI_METHODS: frozenset[str] = frozenset(
    {"gradcam", "gradcam_pp", "saliency", "smoothgrad", "integrated_gradients", "shap", "all"}
)


def predict_single_image(
    image_path: Path,
    config: TrainingConfig,
    xai_method: str = "gradcam",
    output_dir: Path | None = None,
) -> None:
    """Performs inference and XAI heatmap generation on a single leaf image.

    Args:
        image_path: Path to the input leaf image file.
        config: Central TrainingConfig instance with path constants.
        xai_method: XAI method to run. One of: 'gradcam', 'gradcam_pp',
                    'saliency', 'smoothgrad', 'integrated_gradients', 'shap', 'all'.
        output_dir: Optional directory to save heatmap PNG files. Defaults to reports/.
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

    # 1. Load class names
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    # 2. Select device and load model
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Running inference on device: %s", device)

    model = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load and preprocess image
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")

    input_tensor = preprocess_image_for_inference(img_rgb).to(device)

    # 4. Perform prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf_val, pred_idx_tensor = torch.max(probs, dim=1)
        pred_idx = int(pred_idx_tensor.item())
        confidence = float(conf_val.item()) * 100.0

    predicted_class = class_names[pred_idx]

    # 5. Determine output directory
    save_dir = output_dir if output_dir else PROJECT_ROOT / "reports"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 6. Instantiate required XAI explainers
    target_layer = model.get_target_layer_for_gradcam()
    method_lower = xai_method.lower().strip()
    run_all = method_lower == "all"

    # Always instantiate Grad-CAM (default)
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    # Map of method_key → (should_run, explainer_instance_or_factory)
    explainers: dict[str, tuple[bool, object]] = {
        "gradcam":                (True,                                grad_cam),
        "gradcam_pp":             (run_all or method_lower == "gradcam_pp",
                                   GradCAMPlusPlus(model=model, target_layer=target_layer)),
        "saliency":               (run_all or method_lower == "saliency",
                                   SaliencyExplainer(model=model)),
        "smoothgrad":             (run_all or method_lower == "smoothgrad",
                                   SmoothGradExplainer(model=model, num_samples=25)),
        "integrated_gradients":   (run_all or method_lower == "integrated_gradients",
                                   IntegratedGradients(model=model)),
        "shap":                   (run_all or method_lower == "shap",
                                   SHAPExplainer(model=model, grid_size=8)),
    }

    saved_paths: list[str] = []

    for method_key, (should_run, explainer) in explainers.items():
        if not should_run:
            continue

        try:
            logger.info("Generating %s heatmap...", method_key)

            if method_key == "integrated_gradients":
                heatmap = explainer.generate_heatmap(  # type: ignore[union-attr]
                    input_tensor=input_tensor, target_class_idx=pred_idx, steps=20
                )
            else:
                heatmap = explainer.generate_heatmap(  # type: ignore[union-attr]
                    input_tensor=input_tensor, target_class_idx=pred_idx
                )

            overlay_img = overlay_heatmap_on_image(
                original_image=img_rgb, heatmap=heatmap, alpha=0.5
            )

            out_path = save_dir / f"{method_key}_{image_path.stem}.png"
            overlay_img.save(out_path)
            saved_paths.append(str(out_path))
            logger.info("  ✅ Saved: %s", out_path)

        except Exception:
            logger.exception("Failed to generate heatmap for method '%s'.", method_key)

    # 7. Fetch Groq LLM insights
    insights = get_disease_explanation(
        class_name=predicted_class,
        confidence=confidence,
        xai_method=xai_method,
    )

    # 8. Display clean terminal output
    print("\n" + "=" * 68)
    print("   PHYTOSHIELD AI — EXPLAINABLE PLANT DISEASE DIAGNOSIS REPORT")
    print("=" * 68)
    print(f" 📄 Image File         : {image_path.name}")
    print(f" 🌿 Diagnosed Class    : {predicted_class}")
    print(f" 🩺 Disease Title      : {insights['disease_title']}")
    print(f" 🎯 Confidence         : {confidence:.2f}%")
    print(f" 🔬 XAI Method(s)      : {xai_method}")
    print("-" * 68)
    print(" 🗺️  Heatmap Files Saved:")
    for p in saved_paths:
        print(f"      {p}")
    print("-" * 68)
    print(" 🧪 Scientific Explanation:")
    print(f"    {insights['explanation']}")
    print("-" * 68)
    print(" 💊 Treatment & Cure:")
    print(f"    {insights['cure']}")
    print("-" * 68)
    print(" 🛡️  Long-term Prevention:")
    print(f"    {insights['prevention']}")
    print("-" * 68)
    print(" ⚠️  Immediate Precautions:")
    print(f"    {insights['precautions']}")
    print("=" * 68 + "\n")


def main() -> None:
    """CLI parser entry point for PhytoShield AI inference."""
    parser = argparse.ArgumentParser(
        description=(
            "PhytoShield AI — Run plant disease inference with XAI heatmap generation. "
            "Supports 6 attribution methods: gradcam, gradcam_pp, saliency, "
            "smoothgrad, integrated_gradients, shap, or all."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input leaf image file (JPEG/PNG/WEBP).",
    )
    parser.add_argument(
        "--xai-method",
        type=str,
        default="gradcam",
        choices=sorted(_CLI_METHODS),
        help=(
            "XAI attribution method to run. "
            "Use 'all' to generate heatmaps for all 6 methods simultaneously."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save heatmap PNG files. Defaults to reports/.",
    )
    args = parser.parse_args()

    config = TrainingConfig()
    predict_single_image(
        image_path=Path(args.image),
        config=config,
        xai_method=args.xai_method,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
