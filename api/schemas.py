"""API Pydantic Schemas for Explainable AI Plant Disease Detection.

Defines input/output data validation schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """API Health Check response schema."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})


class PredictionResponse(BaseModel):
    """Plant disease prediction and explanation response schema."""
    class_name: str = Field(..., description="Raw model predicted category name")
    disease_title: str = Field(..., description="Clean human-readable disease title")
    confidence: float = Field(..., description="Prediction confidence score percentage (0-100)")
    xai_method: str = Field("gradcam", description="Selected XAI technique name")
    gradcam_heatmap_b64: str = Field(..., description="Base64 encoded Grad-CAM heatmap overlay image string")
    ig_heatmap_b64: str | None = Field(None, description="Base64 encoded Integrated Gradients heatmap overlay")
    shap_heatmap_b64: str | None = Field(None, description="Base64 encoded SHAP heatmap overlay")
    explanation: str = Field(..., description="Detailed disease scientific explanation")
    xai_feature_explanation: str = Field(..., description="Groq LLM explanation of visual features and attribution heatmaps used for prediction")
    cure: str = Field(..., description="Recommended organic and chemical treatments")
    prevention: str = Field(..., description="Long-term agricultural prevention strategies")
    precautions: str = Field(..., description="Immediate agricultural precautions")
