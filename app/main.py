"""Main FastAPI Application Entry Point for Explainable AI Plant Disease Detection.

Configures FastAPI app, CORS middleware, API routes, and model initialization.
"""

import logging
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import load_model_and_metadata, router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads trained model weights and metadata on application startup."""
    logger.info(
        "Initializing PhytoShield AI — 7-method XAI Plant Disease Detection Backend..."
    )
    load_model_and_metadata()
    yield


app = FastAPI(
    title="PhytoShield AI — Explainable Plant Disease Detection API",
    description=(
        "Plant disease classification with a 7-method XAI attribution engine: "
        "Grad-CAM, Grad-CAM++, Vanilla Saliency, SmoothGrad, "
        "Integrated Gradients, SHAP Occlusion, and LIME. "
        "Powered by EfficientNet-B0 (99.48% accuracy) + Groq LLM pathology reports."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint welcoming users."""
    return {
        "message": "Welcome to Explainable AI Plant Disease Detection API",
        "documentation": "/docs",
    }

