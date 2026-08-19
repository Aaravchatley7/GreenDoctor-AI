"""API Pydantic Schemas for Explainable AI Plant Disease Detection.

Defines input/output data validation schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """API Health Check response schema."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})


class PipelineTimings(BaseModel):
    """Monotonic high-resolution execution duration measurements (in seconds)."""
    preprocessing_seconds: float = Field(..., description="Image preprocessing and normalization time in seconds")
    classification_seconds: float = Field(..., description="EfficientNet-B0 forward pass inference time in seconds")
    gradcam_seconds: float = Field(..., description="Grad-CAM convolutional activation map generation time in seconds")
    gradcam_pp_seconds: float | None = Field(None, description="Grad-CAM++ higher-order gradient map generation time in seconds")
    saliency_seconds: float | None = Field(None, description="Vanilla Saliency pixel gradient computation time in seconds")
    smoothgrad_seconds: float | None = Field(None, description="SmoothGrad noise-averaged saliency computation time in seconds")
    ig_seconds: float = Field(..., description="Integrated Gradients 20-step path calculation time in seconds")
    shap_seconds: float = Field(..., description="SHAP 8x8 patch occlusion sensitivity computation time in seconds")
    lime_seconds: float | None = Field(None, description="LIME SLIC superpixel surrogate model fitting time in seconds")
    xai_wall_seconds: float = Field(..., description="Total wall-clock duration for all 7 XAI attribution engines")
    groq_seconds: float = Field(..., description="Groq LLM clinical interpretation API time in seconds")
    total_seconds: float = Field(..., description="Total end-to-end diagnostic pipeline execution time in seconds")



class PredictionResponse(BaseModel):
    """Plant disease prediction and explanation response schema.

    All heatmap fields are base64-encoded PNG data URLs that can be rendered
    directly in an <img> tag: ``src={heatmap_b64}``.
    Fields are None when the corresponding XAI method was not requested.
    """

    class_name: str = Field(..., description="Raw model predicted category name")
    disease_title: str = Field(..., description="Clean human-readable disease title")
    confidence: float = Field(..., description="Prediction confidence score percentage (0-100)")
    xai_method: str = Field(
        "all",
        description="Initial preferred XAI technique or 'all'. All 4 core methods are always generated.",
    )
    # ── Gradient × Activation ─────────────────────────────────────────────────
    gradcam_heatmap_b64: str = Field(
        ..., description="Base64 Grad-CAM heatmap overlay (always generated)"
    )
    gradcam_pp_heatmap_b64: str | None = Field(
        None, description="Base64 Grad-CAM++ heatmap overlay (sharper localization)"
    )
    # ── Pixel-level gradient ──────────────────────────────────────────────────
    saliency_heatmap_b64: str | None = Field(
        None, description="Base64 Vanilla Saliency Map overlay (fastest pixel-level)"
    )
    smoothgrad_heatmap_b64: str | None = Field(
        None, description="Base64 SmoothGrad heatmap overlay (denoised saliency, 25 samples)"
    )
    ig_heatmap_b64: str | None = Field(
        None, description="Base64 Integrated Gradients overlay (axiomatically complete)"
    )
    # ── Perturbation-based ────────────────────────────────────────────────────
    shap_heatmap_b64: str | None = Field(
        None, description="Base64 SHAP occlusion heatmap overlay (game-theory patch scores)"
    )
    lime_heatmap_b64: str | None = Field(
        None, description="Base64 LIME superpixel heatmap (model-agnostic surrogate)"
    )
    # ── LLM Insights ──────────────────────────────────────────────────────────
    explanation: str = Field(..., description="Detailed disease scientific explanation")
    xai_feature_explanation: str = Field(
        ...,
        description="Groq LLM natural-language explanation of XAI visual features",
    )
    cure: str = Field(..., description="Recommended organic and chemical treatments")
    prevention: str = Field(..., description="Long-term agricultural prevention strategies")
    precautions: str = Field(..., description="Immediate agricultural precautions")
    # ── Real High-Resolution Timings ──────────────────────────────────────────
    timings: PipelineTimings = Field(..., description="Real measured high-resolution execution durations")



class XAIExplanationRequest(BaseModel):
    """Request schema for on-demand method-specific XAI LLM explanation."""
    class_name: str = Field(..., description="Raw model predicted category name")
    confidence: float = Field(..., description="Prediction confidence score percentage (0-100)")
    xai_method: str = Field("gradcam", description="Target XAI technique to explain (gradcam, ig, shap, lime, all)")


class XAIExplanationResponse(BaseModel):
    """Response schema for method-specific XAI LLM explanation."""
    xai_method: str = Field(..., description="Target XAI technique explained")
    heading: str = Field(..., description="Context-aware panel heading")
    method_label: str = Field(..., description="Human-friendly method title")
    explanation: str = Field(..., description="Contextual LLM explanation for this specific XAI method")

