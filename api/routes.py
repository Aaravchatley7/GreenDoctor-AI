"""API Routes for Explainable AI Plant Disease Detection.

Implements prediction, Grad-CAM, Integrated Gradients, SHAP heatmap generation, and Groq LLM disease insight endpoints.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from PIL import Image

from api.schemas import HealthCheckResponse, PredictionResponse
from app.services.llm_explainer import get_disease_explanation
from explainability.grad_cam import GradCAM
from explainability.integrated_gradients import IntegratedGradients
from explainability.shap_explainer import SHAPExplainer
from explainability.visualizer import encode_image_to_base64, overlay_heatmap_on_image
from models.efficientnet import build_efficientnet_b0
from preprocessing.transforms import preprocess_image_for_inference

logger = logging.getLogger("api.routes")
router = APIRouter()

# Global state holders for model, class mapping, and XAI instances
MODEL: torch.nn.Module | None = None
CLASS_MAPPING: dict[str, int] = {}
IDX_TO_CLASS: dict[int, str] = {}
GRAD_CAM: GradCAM | None = None
IG_EXPLAINER: IntegratedGradients | None = None
SHAP_EXPLAINER: SHAPExplainer | None = None

if torch.cuda.is_available():
    DEVICE: torch.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE: torch.device = torch.device("mps")
else:
    DEVICE: torch.device = torch.device("cpu")


def load_model_and_metadata() -> None:
    """Loads class mapping, model state, and XAI engines into global scope."""
    global MODEL, CLASS_MAPPING, IDX_TO_CLASS, GRAD_CAM, IG_EXPLAINER, SHAP_EXPLAINER

    project_root = Path(__file__).resolve().parent.parent
    mapping_path = project_root / "data" / "class_mapping.json"
    checkpoint_path = project_root / "models" / "best_model.pth"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            CLASS_MAPPING = json.load(f)
            IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
    else:
        logger.warning("Class mapping file not found at %s.", mapping_path)
        CLASS_MAPPING = {"Tomato_healthy": 0, "Potato___Early_blight": 1, "Potato___Late_blight": 2}
        IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

    num_classes = max(len(CLASS_MAPPING), 1)
    MODEL = build_efficientnet_b0(num_classes=num_classes, pretrained=True)

    if checkpoint_path.exists():
        logger.info("Loading trained weights from %s...", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("Checkpoint '%s' not found. Running with initial weights.", checkpoint_path)

    MODEL.to(DEVICE)
    MODEL.eval()

    target_layer = MODEL.get_target_layer_for_gradcam()
    GRAD_CAM = GradCAM(model=MODEL, target_layer=target_layer)
    IG_EXPLAINER = IntegratedGradients(model=MODEL)
    SHAP_EXPLAINER = SHAPExplainer(model=MODEL, grid_size=8)
    logger.info("Model, Grad-CAM, Integrated Gradients, and SHAP engines initialized successfully.")


@router.get("/health", response_model=HealthCheckResponse)
def health_check() -> dict[str, Any]:
    """Health check endpoint to verify backend service state."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    xai_method: str = Query("gradcam", description="XAI method: 'gradcam', 'integrated_gradients', 'shap', or 'all'"),
) -> dict[str, Any]:
    """Uploads a plant leaf image, predicts disease, generates XAI heatmaps, and fetches Groq LLM insights.

    Args:
        file: Uploaded image file (JPEG, PNG, WEBP).
        xai_method: Selected XAI technique ('gradcam', 'integrated_gradients', 'shap', or 'all').

    Returns:
        PredictionResponse: Prediction, confidence, XAI heatmap base64 strings, and Groq advice.
    """
    if MODEL is None or GRAD_CAM is None or IG_EXPLAINER is None or SHAP_EXPLAINER is None:
        load_model_and_metadata()

    # Validate uploaded content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        logger.error("Failed to read image upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file.",
        ) from exc

    # Preprocess image tensor
    input_tensor = preprocess_image_for_inference(image).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        logits = MODEL(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf_val, pred_idx_tensor = torch.max(probs, dim=1)
        pred_idx = int(pred_idx_tensor.item())
        confidence = float(conf_val.item()) * 100.0

    raw_class_name = IDX_TO_CLASS.get(pred_idx, "Unknown_Class")

    # Generate Grad-CAM heatmap (always default)
    gradcam_heatmap = GRAD_CAM.generate_heatmap(input_tensor=input_tensor, target_class_idx=pred_idx)
    gradcam_overlay = overlay_heatmap_on_image(original_image=image, heatmap=gradcam_heatmap, alpha=0.5)
    gradcam_b64 = encode_image_to_base64(gradcam_overlay)

    ig_b64: str | None = None
    shap_b64: str | None = None

    # Generate additional requested XAI heatmaps
    method_lower = xai_method.lower()
    if method_lower in ["integrated_gradients", "all"]:
        ig_heatmap = IG_EXPLAINER.generate_heatmap(input_tensor=input_tensor, target_class_idx=pred_idx, steps=20)
        ig_overlay = overlay_heatmap_on_image(original_image=image, heatmap=ig_heatmap, alpha=0.5)
        ig_b64 = encode_image_to_base64(ig_overlay)

    if method_lower in ["shap", "all"]:
        shap_heatmap = SHAP_EXPLAINER.generate_heatmap(input_tensor=input_tensor, target_class_idx=pred_idx)
        shap_overlay = overlay_heatmap_on_image(original_image=image, heatmap=shap_heatmap, alpha=0.5)
        shap_b64 = encode_image_to_base64(shap_overlay)

    # Fetch Groq LLM insights including XAI visual feature attribution
    llm_insights = get_disease_explanation(
        class_name=raw_class_name,
        confidence=confidence,
        xai_method=xai_method,
    )

    return {
        "class_name": raw_class_name,
        "disease_title": llm_insights["disease_title"],
        "confidence": round(confidence, 2),
        "xai_method": xai_method,
        "gradcam_heatmap_b64": gradcam_b64,
        "ig_heatmap_b64": ig_b64,
        "shap_heatmap_b64": shap_b64,
        "explanation": llm_insights["explanation"],
        "xai_feature_explanation": llm_insights["xai_feature_explanation"],
        "cure": llm_insights["cure"],
        "prevention": llm_insights["prevention"],
        "precautions": llm_insights["precautions"],
    }
