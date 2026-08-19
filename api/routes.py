"""API Routes for Explainable AI Plant Disease Detection.

Implements prediction with a 7-method XAI attribution engine:
  • Grad-CAM             — class-discriminative coarse region heatmap
  • Grad-CAM++           — second-order gradient weighting, sharper localization
  • Vanilla Saliency     — fastest pixel-level |∂y/∂x| attribution
  • SmoothGrad           — noise-averaged saliency for cleaner pixel maps
  • Integrated Gradients — axiomatically complete path-integral attribution
  • SHAP Occlusion       — game-theory patch-importance maps
  • LIME                 — model-agnostic superpixel surrogate (explicit request only)

All methods return normalized [0,1] np.ndarray heatmaps and are rendered
as COLORMAP_JET overlays via the shared visualizer utility.
"""

import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from PIL import Image

from api.schemas import (
    HealthCheckResponse,
    PipelineTimings,
    PredictionResponse,
    XAIExplanationRequest,
    XAIExplanationResponse,
)

from app.services.llm_explainer import get_disease_explanation, get_xai_specific_explanation
from explainability.grad_cam import GradCAM
from explainability.grad_cam_plus_plus import GradCAMPlusPlus
from explainability.integrated_gradients import IntegratedGradients
from explainability.lime_explainer import LIMEExplainer
from explainability.saliency import SaliencyExplainer, SmoothGradExplainer
from explainability.shap_explainer import SHAPExplainer
from explainability.visualizer import encode_all_heatmaps, overlay_heatmap_on_image, encode_image_to_base64
from models.efficientnet import build_efficientnet_b0
from preprocessing.transforms import preprocess_image_for_inference

logger = logging.getLogger("api.routes")
router = APIRouter()

# ── Global engine holders ────────────────────────────────────────────────────
MODEL: torch.nn.Module | None = None
CLASS_MAPPING: dict[str, int] = {}
IDX_TO_CLASS: dict[int, str] = {}

GRAD_CAM: GradCAM | None = None
GRAD_CAM_PP: GradCAMPlusPlus | None = None
SALIENCY_EXPLAINER: SaliencyExplainer | None = None
SMOOTH_GRAD_EXPLAINER: SmoothGradExplainer | None = None
IG_EXPLAINER: IntegratedGradients | None = None
SHAP_EXPLAINER: SHAPExplainer | None = None
LIME_EXPLAINER: LIMEExplainer | None = None

# Methods included in xai_method=all (excludes LIME due to ~2–5s latency)
_FAST_METHODS: frozenset[str] = frozenset(
    {"gradcam", "gradcam_pp", "saliency", "smoothgrad", "integrated_gradients", "shap"}
)

_VALID_METHODS: frozenset[str] = _FAST_METHODS | {"lime", "all"}

if torch.cuda.is_available():
    DEVICE: torch.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE: torch.device = torch.device("mps")
else:
    DEVICE: torch.device = torch.device("cpu")


def load_model_and_metadata() -> None:
    """Loads class mapping, model weights, and all 7 XAI engines into global scope."""
    global MODEL, CLASS_MAPPING, IDX_TO_CLASS
    global GRAD_CAM, GRAD_CAM_PP
    global SALIENCY_EXPLAINER, SMOOTH_GRAD_EXPLAINER
    global IG_EXPLAINER, SHAP_EXPLAINER, LIME_EXPLAINER


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

    # ── Initialise XAI engines ───────────────────────────────────────────────
    target_layer = MODEL.get_target_layer_for_gradcam()

    GRAD_CAM              = GradCAM(model=MODEL, target_layer=target_layer)
    GRAD_CAM_PP           = GradCAMPlusPlus(model=MODEL, target_layer=target_layer)
    SALIENCY_EXPLAINER    = SaliencyExplainer(model=MODEL)
    SMOOTH_GRAD_EXPLAINER = SmoothGradExplainer(model=MODEL, num_samples=25, noise_level=0.1)
    IG_EXPLAINER          = IntegratedGradients(model=MODEL)
    SHAP_EXPLAINER        = SHAPExplainer(model=MODEL, grid_size=8)
    LIME_EXPLAINER        = LIMEExplainer(model=MODEL, num_samples=100, num_features=8)

    logger.info(
        "All 7 XAI engines initialized: Grad-CAM, Grad-CAM++, Saliency, "
        "SmoothGrad, Integrated Gradients, SHAP, LIME."
    )


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
    xai_method: str = Query(
        "all",
        description=(
            "Preferred initial XAI view. All 4 attribution methods (Grad-CAM, IG, SHAP, LIME) "
            "are always automatically generated in parallel."
        ),
    ),
) -> dict[str, Any]:
    """Uploads a leaf image, runs disease classification, computes ALL 4 XAI attribution engines,
    and measures actual high-resolution execution duration.

    Args:
        file: Uploaded image file (JPEG, PNG, WEBP).
        xai_method: Initial selected XAI method for display preference.

    Returns:
        PredictionResponse: Prediction, confidence, all 4 XAI heatmaps, Groq LLM insights,
        and high-resolution monotonic PipelineTimings.

    Raises:
        HTTPException 400: Invalid file type or corrupted image.
        HTTPException 422: Unknown xai_method value.
        HTTPException 503: Model not loaded.
    """
    t_pipe_start = time.perf_counter()

    if MODEL is None:
        load_model_and_metadata()

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image.",
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        logger.error("Failed to read image upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file.",
        ) from exc

    # ── Stage 1: Image Preprocessing ──────────────────────────────────────────
    t_pre_0 = time.perf_counter()
    input_tensor = preprocess_image_for_inference(image).to(DEVICE)
    dur_pre = time.perf_counter() - t_pre_0

    # ── Stage 2: Disease Classification ───────────────────────────────────────
    t_cls_0 = time.perf_counter()
    with torch.no_grad():
        logits = MODEL(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf_val, pred_idx_tensor = torch.max(probs, dim=1)
        pred_idx = int(pred_idx_tensor.item())
        confidence = float(conf_val.item()) * 100.0
    dur_cls = time.perf_counter() - t_cls_0

    raw_class_name = IDX_TO_CLASS.get(pred_idx, "Unknown_Class")

    # ── Stage 3: XAI Attribution Generation (ALL Methods Generated) ───────────
    t_xai_wall_0 = time.perf_counter()
    heatmaps: dict[str, Any] = {}

    # 1. Grad-CAM (Convolutional Activation)
    t_g0 = time.perf_counter()
    heatmaps["gradcam"] = GRAD_CAM.generate_heatmap(
        input_tensor=input_tensor, target_class_idx=pred_idx
    )
    dur_gradcam = time.perf_counter() - t_g0

    # 2. Grad-CAM++ (Second-Order Gradient Map)
    t_gpp0 = time.perf_counter()
    heatmaps["gradcam_pp"] = GRAD_CAM_PP.generate_heatmap(input_tensor=input_tensor, target_class_idx=pred_idx)
    dur_gradcam_pp = time.perf_counter() - t_gpp0

    # 3. Vanilla Saliency (Pixel-Level Gradient |∂y/∂x|)
    t_sal0 = time.perf_counter()
    heatmaps["saliency"] = SALIENCY_EXPLAINER.generate_heatmap(input_tensor=input_tensor, target_class_idx=pred_idx)
    dur_saliency = time.perf_counter() - t_sal0

    # 4. SmoothGrad (Noise-Averaged Saliency, 25 samples)
    t_sg0 = time.perf_counter()
    heatmaps["smoothgrad"] = SMOOTH_GRAD_EXPLAINER.generate_heatmap(input_tensor=input_tensor, target_class_idx=pred_idx)
    dur_smoothgrad = time.perf_counter() - t_sg0

    # 5. Integrated Gradients (Axiomatic Path Integration, 20 steps)
    t_ig0 = time.perf_counter()
    heatmaps["ig"] = IG_EXPLAINER.generate_heatmap(
        input_tensor=input_tensor, target_class_idx=pred_idx, steps=20
    )
    dur_ig = time.perf_counter() - t_ig0

    # 6. SHAP (Shapley Patch Occlusion, 8x8 grid)
    t_shap0 = time.perf_counter()
    heatmaps["shap"] = SHAP_EXPLAINER.generate_heatmap(
        input_tensor=input_tensor, target_class_idx=pred_idx
    )
    dur_shap = time.perf_counter() - t_shap0

    # 7. LIME (SLIC Superpixel Surrogate)
    t_lime0 = time.perf_counter()
    try:
        heatmaps["lime"] = LIME_EXPLAINER.generate_heatmap(
            input_tensor=input_tensor,
            target_class_idx=pred_idx,
            original_image=image,
        )
        dur_lime = time.perf_counter() - t_lime0
    except Exception as exc:
        logger.exception("LIME attribution generation failed: %s", exc)
        heatmaps["lime"] = None
        dur_lime = None

    dur_xai_wall = time.perf_counter() - t_xai_wall_0

    # ── Batch encode all heatmaps → base64 ───────────────────────────────────
    encoded = encode_all_heatmaps(original_image=image, heatmaps=heatmaps, alpha=0.5)

    # ── Stage 4: Groq LLM Insights ───────────────────────────────────────────
    t_groq0 = time.perf_counter()
    llm_insights = get_disease_explanation(
        class_name=raw_class_name,
        confidence=confidence,
        xai_method=xai_method,
    )
    dur_groq = time.perf_counter() - t_groq0

    # ── Total Pipeline Duration ──────────────────────────────────────────────
    dur_total = time.perf_counter() - t_pipe_start

    timings = PipelineTimings(
        preprocessing_seconds=round(dur_pre, 3),
        classification_seconds=round(dur_cls, 3),
        gradcam_seconds=round(dur_gradcam, 3),
        gradcam_pp_seconds=round(dur_gradcam_pp, 3),
        saliency_seconds=round(dur_saliency, 3),
        smoothgrad_seconds=round(dur_smoothgrad, 3),
        ig_seconds=round(dur_ig, 3),
        shap_seconds=round(dur_shap, 3),
        lime_seconds=round(dur_lime, 3) if dur_lime is not None else None,
        xai_wall_seconds=round(dur_xai_wall, 3),
        groq_seconds=round(dur_groq, 3),
        total_seconds=round(dur_total, 3),
    )


    logger.info(
        "Pipeline executed: Pre=%.3fs, Cls=%.3fs, CAM=%.3fs, IG=%.3fs, SHAP=%.3fs, LIME=%ss, Groq=%.3fs, Total=%.3fs",
        dur_pre, dur_cls, dur_gradcam, dur_ig, dur_shap, f"{dur_lime:.3f}" if dur_lime else "N/A", dur_groq, dur_total
    )

    return {
        "class_name":              raw_class_name,
        "disease_title":           llm_insights["disease_title"],
        "confidence":              round(confidence, 2),
        "xai_method":              xai_method,
        "gradcam_heatmap_b64":     encoded["gradcam"],
        "gradcam_pp_heatmap_b64":  encoded.get("gradcam_pp"),
        "saliency_heatmap_b64":    encoded.get("saliency"),
        "smoothgrad_heatmap_b64":  encoded.get("smoothgrad"),
        "ig_heatmap_b64":          encoded.get("ig"),
        "shap_heatmap_b64":        encoded.get("shap"),
        "lime_heatmap_b64":        encoded.get("lime"),
        "explanation":             llm_insights["explanation"],
        "xai_feature_explanation": llm_insights["xai_feature_explanation"],
        "cure":                    llm_insights["cure"],
        "prevention":              llm_insights["prevention"],
        "precautions":             llm_insights["precautions"],
        "timings":                 timings,
    }


@router.post("/explain-xai", response_model=XAIExplanationResponse)
async def explain_xai_method(payload: XAIExplanationRequest) -> dict[str, str]:
    """Generates an on-demand, method-tailored Groq LLM explanation for a specific XAI technique.

    Args:
        payload: XAIExplanationRequest containing class_name, confidence, and target xai_method.

    Returns:
        XAIExplanationResponse: Context-aware heading, human-friendly label, and explanation text.
    """
    return get_xai_specific_explanation(
        class_name=payload.class_name,
        confidence=payload.confidence,
        xai_method=payload.xai_method,
    )

