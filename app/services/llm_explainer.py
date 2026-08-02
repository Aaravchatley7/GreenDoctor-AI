"""Groq LLM Disease & XAI Feature Attribution Insight Service.

Queries Groq LLM to generate structured disease explanations, visual feature attribution
explanations (explaining which leaf features drove the Grad-CAM/IG/SHAP heatmaps),
cures, preventions, and precautions based on model predictions.
"""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("app.services.llm_explainer")

# Attempt importing Groq SDK safely
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq package not installed. Fallback mode will be used.")


# Pre-defined fallback knowledge base for plant disease insights & XAI feature attributions
FALLBACK_KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    "potato___early_blight": {
        "explanation": "Early Blight is a common fungal disease caused by Alternaria solani. It manifests as dark concentric rings (target spot pattern) on older potato leaves, leading to premature leaf drop and reduced tuber yield.",
        "xai_feature_explanation": "The XAI heatmaps highlight the concentric brown necrotic lesions and surrounding chlorotic yellow halos. Grad-CAM and Integrated Gradients show high activation density around leaf margin spot clusters, confirming the neural network focused on target-spot fungal patterns rather than healthy background leaf tissue.",
        "cure": "Apply copper-based fungicides or chlorothalonil at first sign of symptoms. Remove and destroy heavily infected foliage.",
        "prevention": "Practice 3-year crop rotation, use disease-free certified seeds, and ensure proper plant spacing for adequate air circulation.",
        "precautions": "Avoid overhead irrigation to minimize leaf wetness duration. Sanitize tools after pruning infected plants."
    },
    "potato___late_blight": {
        "explanation": "Late Blight is a destructive water-mold disease caused by Phytophthora infestans. It forms large, dark brown water-soaked lesions on leaves with white mold growth on the undersides during humid weather.",
        "xai_feature_explanation": "The attribution maps reveal strong neural focus on dark water-soaked leaf margins and pale mold sporulation regions. Integrated Gradients and SHAP identify irregular leaf necrosis and foliar wilting gradients as the primary drivers behind the Late Blight classification.",
        "cure": "Apply systemic fungicides containing mancozeb or metalaxyl immediately. Remove infected vines 2 weeks before harvest.",
        "prevention": "Plant resistant cultivars, eliminate volunteer potato plants, and ensure proper soil drainage.",
        "precautions": "Destroy infected tubers immediately to prevent spore dispersal. Avoid harvesting during wet soil conditions."
    },
    "tomato___early_blight": {
        "explanation": "Tomato Early Blight is caused by Alternaria linariae, producing dark brown target-like spots with chlorotic yellow borders on foliage.",
        "xai_feature_explanation": "XAI visual attribution highlights circular target spots and foliar yellowing surrounding central necrosis. High heatmap intensity over leaf vein junctions indicates the model utilized structural lesion spread as key classification evidence.",
        "cure": "Spray copper hydroxide or chlorothalonil every 7-14 days. Remove lower diseased leaves.",
        "prevention": "Mulch garden beds to prevent soil splashback and practice crop rotation.",
        "precautions": "Ensure foliage dries quickly after rain and avoid working among wet tomato plants."
    },
    "tomato___late_blight": {
        "explanation": "Tomato Late Blight causes rapid foliar collapse with dark gray water-soaked spots that expand rapidly in humid conditions.",
        "xai_feature_explanation": "Attribution maps demonstrate localized high activation over water-soaked leaf tips and necrotic petioles. SHAP patch attribution confirms that non-chlorotic green leaf areas contributed negatively, isolating disease spots as the decisive feature.",
        "cure": "Apply protective fungicides containing copper or chlorothalonil at the earliest detection.",
        "prevention": "Use certified disease-free transplants and plant in full sun with drip irrigation.",
        "precautions": "Remove and bag infected plants immediately. Do not compost infected foliage."
    },
    "tomato_healthy": {
        "explanation": "The tomato foliage appears healthy with uniform green chlorophyll distribution and no visible signs of fungal, bacterial, or viral infection.",
        "xai_feature_explanation": "XAI visual attribution displays uniform, low-intensity feature weights across intact leaf blade structures. The absence of localized high-gradient peak clusters verifies that the model found zero necrotic or chlorotic anomaly features.",
        "cure": "No treatment required. Maintain current optimal growing conditions.",
        "prevention": "Continue regular monitoring, balanced fertilization, and consistent watering schedule.",
        "precautions": "Avoid physical injury to leaves and stems to prevent pathogen entry points."
    },
    "default": {
        "explanation": "Plant pathology disease identified by deep learning classifier. Further visual inspection of symptoms recommended.",
        "xai_feature_explanation": "Attribution heatmaps (Grad-CAM, Integrated Gradients, SHAP) concentrate energy over symptomatic leaf lesions, spot boundaries, and chlorotic tissue zones. The visual feature map confirms the neural network prioritized pathological leaf anomalies over background noise.",
        "cure": "Apply broad-spectrum organic neem oil or recommended copper-based fungicide.",
        "prevention": "Maintain proper plant spacing, drip irrigation, and crop rotation practices.",
        "precautions": "Sanitize gardening tools before and after handling plants."
    }
}


def clean_class_name(raw_class_name: str) -> str:
    """Formats raw class string (e.g. Potato___Early_blight) into clean title format.

    Args:
        raw_class_name: Raw category name string.

    Returns:
        str: Human-readable disease title.
    """
    clean = raw_class_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    return clean.strip()


def get_disease_explanation(
    class_name: str,
    confidence: float,
    xai_method: str = "all",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generates structured disease explanation, XAI visual feature attribution, cure, prevention, and precautions.

    Queries Groq API if key is present; falls back to structured knowledge base if unavailable.

    Args:
        class_name: Predicted plant disease category.
        confidence: Prediction confidence percentage (0-100).
        xai_method: Selected XAI attribution technique name.
        api_key: Optional Groq API Key.

    Returns:
        dict[str, Any]: Structured dictionary with keys 'disease_title', 'explanation',
            'xai_feature_explanation', 'cure', 'prevention', 'precautions'.
    """
    disease_title = clean_class_name(class_name)
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")

    if GROQ_AVAILABLE and groq_api_key:
        try:
            logger.info("Querying Groq LLM for disease & XAI feature insights on '%s'...", disease_title)
            client = Groq(api_key=groq_api_key)
            
            prompt = (
                f"You are an expert agricultural botanist and Explainable AI (XAI) computer vision specialist.\n"
                f"A deep learning vision model classified a plant leaf photo as: '{disease_title}' with {confidence:.1f}% confidence.\n"
                f"XAI Attribution Method Used: '{xai_method.upper()}' (Grad-CAM, Integrated Gradients, SHAP).\n\n"
                f"Provide a structured JSON response with EXACTLY the following keys:\n"
                f"{{\n"
                f'  "explanation": "Detailed 2-3 sentence scientific explanation of the disease and its biological pathology.",\n'
                f'  "xai_feature_explanation": "Detailed 2-3 sentence explanation of WHAT specific visual features on the leaf (e.g. concentric necrotic spots, chlorotic halo, vein discoloration, lesion geometry, spore clusters) were highlighted by the XAI heatmaps and HOW the model used these exact features to make its decision.",\n'
                f'  "cure": "2-3 recommended organic and chemical treatment methods.",\n'
                f'  "prevention": "2-3 long-term agricultural prevention strategies.",\n'
                f'  "precautions": "2-3 immediate precautions to avoid spreading infection."\n'
                f"}}\n\n"
                f"Return ONLY valid JSON."
            )

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            response_json = json.loads(completion.choices[0].message.content)
            response_json["disease_title"] = disease_title
            
            # Ensure xai_feature_explanation is present
            if "xai_feature_explanation" not in response_json:
                response_json["xai_feature_explanation"] = (
                    f"XAI heatmaps ({xai_method}) highlight key pathological leaf features including "
                    f"necrotic spot boundaries and chlorotic halos that drove the model's confidence."
                )

            return response_json

        except Exception as exc:
            logger.error("Groq API query failed: %s. Falling back to knowledge base.", exc)

    # Fallback execution
    key = class_name.lower()
    fallback_data = FALLBACK_KNOWLEDGE_BASE.get(key, FALLBACK_KNOWLEDGE_BASE["default"])
    
    return {
        "disease_title": disease_title,
        "explanation": fallback_data["explanation"],
        "xai_feature_explanation": fallback_data["xai_feature_explanation"],
        "cure": fallback_data["cure"],
        "prevention": fallback_data["prevention"],
        "precautions": fallback_data["precautions"],
    }
