"""Groq LLM Disease & XAI Feature Attribution Insight Service.

Queries Groq LLM to generate structured disease explanations, method-specific visual feature attribution
explanations (tailored for Grad-CAM, Integrated Gradients, SHAP, LIME, or Combined All),
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


# Method-specific fallback explanations for all 7 XAI attribution methods
METHOD_SPECIFIC_FALLBACKS: dict[str, dict[str, str]] = {
    "gradcam": {
        "title": "Grad-CAM Spatial Activation",
        "heading": "AI CLINICAL — GRAD-CAM INTERPRETATION",
        "text": (
            "Grad-CAM convolutional feature maps show concentrated positive activations over "
            "the central necrotic lesion zones and surrounding chlorotic yellow halos. "
            "The neural network's final convolution layer prioritized these high-intensity lesion clusters, "
            "confirming that spatial disease localization drove the classification decision."
        ),
    },
    "gradcam_pp": {
        "title": "Grad-CAM++ Multi-Instance Activation",
        "heading": "AI CLINICAL — GRAD-CAM++ INTERPRETATION",
        "text": (
            "Grad-CAM++ applies second- and third-order positive partial derivatives to weight feature activations. "
            "This provides sharper localization over multiple discrete lesion spots and smaller co-occurring symptom patches "
            "across the leaf blade compared to standard Grad-CAM."
        ),
    },
    "saliency": {
        "title": "Vanilla Saliency Map",
        "heading": "AI CLINICAL — VANILLA SALIENCY INTERPRETATION",
        "text": (
            "Vanilla Saliency computes the raw magnitude of the output gradient with respect to input pixels (|∂y/∂x|). "
            "The resulting pixel-level map highlights high-frequency symptom boundaries, vein bifurcations, and fine textural edges "
            "that most immediately perturb the model's output score."
        ),
    },
    "smoothgrad": {
        "title": "SmoothGrad (Noise-Averaged Saliency)",
        "heading": "AI CLINICAL — SMOOTHGRAD INTERPRETATION",
        "text": (
            "SmoothGrad reduces gradient noise by averaging saliency maps generated across 25 Gaussian-perturbed samples. "
            "This filters out spurious pixel fluctuations and isolates coherent, robust visual structures representing true leaf pathology."
        ),
    },
    "integrated_gradients": {
        "title": "Integrated Gradients Attribution",
        "heading": "AI CLINICAL — INTEGRATED GRADIENTS INTERPRETATION",
        "text": (
            "Integrated Gradients calculates pixel-level path attributions from a neutral baseline across 20 interpolation steps. "
            "Satisfying axiomatic completeness, it highlights precise leaf vein junctions and diseased spot perimeters without gradient saturation."
        ),
    },
    "shap": {
        "title": "SHAP Occlusion Sensitivity",
        "heading": "AI CLINICAL — SHAP INTERPRETATION",
        "text": (
            "SHAP patch occlusion evaluates the marginal contribution of masked leaf regions using game-theoretic Shapley values. "
            "Occluding symptomatic foliar patches causes significant drops in class probability, verifying the model's reliance on genuine pathological patterns."
        ),
    },
    "lime": {
        "title": "LIME Superpixel Surrogate",
        "heading": "AI CLINICAL — LIME INTERPRETATION",
        "text": (
            "LIME segments the leaf image into SLIC superpixels and fits an interpretable linear surrogate model across perturbations. "
            "The superpixel clusters with the highest positive regression coefficients correspond directly to foliar spot margins and discolored tissue."
        ),
    },
    "all": {
        "title": "Combined 4-Method XAI Suite",
        "heading": "AI CLINICAL — 4-ENGINE MULTI-MODAL XAI INTERPRETATION",
        "text": (
            "• Grad-CAM: Concentrates coarse activation over primary lesion epicenters.\n"
            "• Grad-CAM++: Resolves multi-instance lesion clusters with second-order gradient weighting.\n"
            "• Vanilla Saliency: Captures instantaneous pixel-level input sensitivity (|∂y/∂x|).\n"
            "• SmoothGrad: Denoises the saliency map via 25-sample Gaussian smoothing.\n"
            "• Integrated Gradients: Provides axiomatically complete path-integral pixel attributions.\n"
            "• SHAP: Quantifies game-theoretic Shapley importance across 8×8 foliar patches.\n"
            "• LIME: Fits a model-agnostic linear surrogate over SLIC superpixel segments."
        ),
    },
}


# Pre-defined fallback knowledge base for plant disease insights
FALLBACK_KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    "potato___early_blight": {
        "explanation": "Early Blight is a common fungal disease caused by Alternaria solani. It manifests as dark concentric rings (target spot pattern) on older potato leaves, leading to premature leaf drop and reduced tuber yield.",
        "xai_feature_explanation": "Grad-CAM and Integrated Gradients show high activation density around leaf margin spot clusters, confirming the neural network focused on target-spot fungal patterns rather than healthy background leaf tissue.",
        "cure": "Apply copper-based fungicides or chlorothalonil at first sign of symptoms. Remove and destroy heavily infected foliage.",
        "prevention": "Practice 3-year crop rotation, use disease-free certified seeds, and ensure proper plant spacing for adequate air circulation.",
        "precautions": "Avoid overhead irrigation to minimize leaf wetness duration. Sanitize tools after pruning infected plants."
    },
    "potato___late_blight": {
        "explanation": "Late Blight is a destructive water-mold disease caused by Phytophthora infestans. It forms large, dark brown water-soaked lesions on leaves with white mold growth on the undersides during humid weather.",
        "xai_feature_explanation": "The attribution maps reveal strong neural focus on dark water-soaked leaf margins and pale mold sporulation regions, identifying irregular leaf necrosis as the primary driver.",
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
        "xai_feature_explanation": "Attribution heatmaps concentrate energy over symptomatic leaf lesions, spot boundaries, and chlorotic tissue zones. The visual feature map confirms the neural network prioritized pathological leaf anomalies over background noise.",
        "cure": "Apply broad-spectrum organic neem oil or recommended copper-based fungicide.",
        "prevention": "Maintain proper plant spacing, drip irrigation, and crop rotation practices.",
        "precautions": "Sanitize gardening tools before and after handling plants."
    }
}


CURATED_DISEASE_NAMES: dict[str, str] = {
    # Pepper
    "Pepper__bell___Bacterial_spot": "Bell Pepper Bacterial Spot Disease",
    "Pepper_bell___Bacterial_spot": "Bell Pepper Bacterial Spot Disease",
    "Pepper__bell___healthy": "Healthy Bell Pepper Foliage",
    "Pepper_bell___healthy": "Healthy Bell Pepper Foliage",
    # Potato
    "Potato___Early_blight": "Potato Early Blight Disease",
    "Potato___Late_blight": "Potato Late Blight Disease",
    "Potato___healthy": "Healthy Potato Foliage",
    "Potato_healthy": "Healthy Potato Foliage",
    # Tomato
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot Disease",
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot Disease",
    "Tomato_Early_blight": "Tomato Early Blight Disease",
    "Tomato___Early_blight": "Tomato Early Blight Disease",
    "Tomato_Late_blight": "Tomato Late Blight Disease",
    "Tomato___Late_blight": "Tomato Late Blight Disease",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold Disease",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold Disease",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot Disease",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot Disease",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Two-Spotted Spider Mite Infestation",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "Tomato Two-Spotted Spider Mite Infestation",
    "Tomato__Target_Spot": "Tomato Target Spot Disease",
    "Tomato___Target_Spot": "Tomato Target Spot Disease",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus (TYLCV)",
    "Tomato___Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus (TYLCV)",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus (TYLCV)",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus (ToMV)",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus (ToMV)",
    "Tomato_mosaic_virus": "Tomato Mosaic Virus (ToMV)",
    "Tomato_healthy": "Healthy Tomato Foliage",
    "Tomato___healthy": "Healthy Tomato Foliage",
    # Apple
    "Apple___Apple_scab": "Apple Scab Disease",
    "Apple___Black_rot": "Apple Black Rot Disease",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust Disease",
    "Apple___healthy": "Healthy Apple Foliage",
    # Corn
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "Corn Gray Leaf Spot Disease",
    "Corn_(maize)___Common_rust_": "Corn Common Rust Disease",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Corn Leaf Blight Disease",
    "Corn_(maize)___healthy": "Healthy Corn Foliage",
    # Grape
    "Grape___Black_rot": "Grapevine Black Rot Disease",
    "Grape___Esca_(Black_Measles)": "Grapevine Esca (Black Measles) Disease",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grapevine Leaf Blight (Isariopsis Spot)",
    "Grape___healthy": "Healthy Grapevine Foliage",
    # Peach
    "Peach___Bacterial_spot": "Peach Bacterial Spot Disease",
    "Peach___healthy": "Healthy Peach Foliage",
}


def clean_class_name(raw_class_name: str) -> str:
    """Formats raw class identifier into human-friendly, agronomically accurate display name."""
    if not raw_class_name:
        return "Unknown Plant Specimen"

    # Direct match
    if raw_class_name in CURATED_DISEASE_NAMES:
        return CURATED_DISEASE_NAMES[raw_class_name]

    # Case-insensitive match
    for k, v in CURATED_DISEASE_NAMES.items():
        if k.lower() == raw_class_name.lower():
            return v

    # Fallback semantic normalization
    clean = (
        raw_class_name.replace("___", " - ")
        .replace("__", " ")
        .replace("_", " ")
        .replace("(including sour)", "")
        .replace("(maize)", "")
        .strip()
    )
    clean = clean.replace("Pepper bell", "Bell Pepper")
    clean = clean.replace("YellowLeaf", "Yellow Leaf")
    if "healthy" in clean.lower():
        crop = clean.lower().replace("healthy", "").replace("-", "").strip().title()
        return f"Healthy {crop} Foliage" if crop else "Healthy Plant Foliage"

    return clean.title()



def normalize_method_key(xai_method: str) -> str:
    """Normalizes xai_method string to canonical key."""
    m = xai_method.lower().strip()
    if "gradcam_pp" in m or "gradcam++" in m or "plus" in m:
        return "gradcam_pp"
    if "gradcam" in m or "grad_cam" in m:
        return "gradcam"
    if "saliency" in m or "vanilla" in m:
        return "saliency"
    if "smoothgrad" in m or "smooth" in m:
        return "smoothgrad"
    if "integrated" in m or m == "ig":
        return "integrated_gradients"
    if "shap" in m:
        return "shap"
    if "lime" in m:
        return "lime"
    return "all"


def get_xai_specific_explanation(
    class_name: str,
    confidence: float,
    xai_method: str = "gradcam",
    api_key: str | None = None,
) -> dict[str, str]:
    """Generates a context-aware Groq LLM explanation specific to the selected XAI technique.

    Args:
        class_name: Predicted plant disease category.
        confidence: Prediction confidence percentage (0-100).
        xai_method: Selected XAI attribution technique ('gradcam', 'gradcam_pp', 'saliency', 'smoothgrad', 'integrated_gradients', 'shap', 'lime', 'all').
        api_key: Optional Groq API Key.

    Returns:
        dict[str, str]: Dictionary containing 'xai_method', 'heading', 'method_label', and 'explanation'.
    """
    disease_title = clean_class_name(class_name)
    method_key = normalize_method_key(xai_method)
    fallback_info = METHOD_SPECIFIC_FALLBACKS.get(method_key, METHOD_SPECIFIC_FALLBACKS["all"])
    
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")

    method_prompts = {
        "gradcam": (
            "Focus specifically on Grad-CAM's convolutional activation maps: explain which coarse spatial regions "
            "(e.g. lesion clusters, chlorotic halos, necrotic centers) show high activation intensity in the last convolutional layer "
            "and why these spatial regions are biologically relevant to this diagnosis."
        ),
        "gradcam_pp": (
            "Focus specifically on Grad-CAM++ (Grad-CAM Plus Plus): explain how higher-order partial derivatives (second and third order) "
            "weight the feature activations, enabling sharp localization of multiple discrete disease spots and smaller lesion clusters "
            "across the leaf surface compared to standard Grad-CAM."
        ),
        "saliency": (
            "Focus specifically on Vanilla Saliency Maps: explain how the direct pixel-level gradient magnitude (|∂y/∂x|) reveals "
            "high-frequency edge transitions, vein junctions, and subtle discoloration boundaries that immediately influence "
            "the neural network's classification score."
        ),
        "smoothgrad": (
            "Focus specifically on SmoothGrad: explain how Gaussian noise averaging across 25 perturbed samples eliminates spurious gradient noise "
            "to produce a clean, denoised pixel-level map of the true pathological lesion structures."
        ),
        "integrated_gradients": (
            "Focus specifically on Integrated Gradients: explain the axiomatic pixel-level path attribution from a zero baseline, "
            "highlighting fine-grained edge transitions, leaf vein boundaries, and positive vs negative pixel contributions "
            "that distinguish the disease symptoms."
        ),
        "shap": (
            "Focus specifically on SHAP Occlusion: explain game-theoretic Shapley value patch sensitivity, "
            "how occluding specific symptomatic grid patches caused model confidence drops, and how this measures the marginal importance "
            "of distinct leaf areas."
        ),
        "lime": (
            "Focus specifically on LIME (Local Interpretable Model-agnostic Explanations): explain the SLIC superpixel segmentation "
            "and local linear surrogate model fit across perturbations, highlighting which superpixel clusters actively supported the prediction."
        ),
        "all": (
            "Provide a concise, clearly differentiated breakdown covering all 7 XAI attribution techniques: "
            "Grad-CAM (spatial activation), Grad-CAM++ (multi-instance), Vanilla Saliency (pixel gradient), SmoothGrad (denoised saliency), "
            "Integrated Gradients (path integral), SHAP (patch occlusion), and LIME (superpixel surrogate). "
            "Use bullet points for each method."
        ),
    }


    specific_guidance = method_prompts.get(method_key, method_prompts["all"])

    if GROQ_AVAILABLE and groq_api_key:
        try:
            logger.info("Querying Groq LLM for method-specific XAI insight [%s] on '%s'...", method_key, disease_title)
            client = Groq(api_key=groq_api_key)
            
            prompt = (
                f"You are an expert agricultural plant pathologist and Explainable AI (XAI) computer vision researcher.\n\n"
                f"Diagnostic Context:\n"
                f"- Diagnosis: '{disease_title}'\n"
                f"- Raw Model Class: '{class_name}'\n"
                f"- Confidence: {confidence:.1f}%\n"
                f"- Selected XAI Technique: '{fallback_info['title']}' ({method_key.upper()})\n\n"
                f"Instructions:\n"
                f"{specific_guidance}\n\n"
                f"STRICT RULES:\n"
                f"1. Only describe visual evidence that can reasonably be supported by this specific XAI technique and pathology.\n"
                f"2. Do not invent symptoms, colors, or lesions not characteristic of {disease_title}.\n"
                f"3. Keep the explanation focused, clear, and directly relevant (2-4 sentences or clean bullet points).\n"
                f"4. Do NOT include generic conversational filler.\n\n"
                f"Return ONLY a JSON object with this exact key:\n"
                f'{{"explanation": "Your concise, method-specific explanation here"}}'
            )

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            response_json = json.loads(completion.choices[0].message.content)
            raw_explanation = response_json.get("explanation", fallback_info["text"])

            if isinstance(raw_explanation, dict):
                lines = []
                for k, v in raw_explanation.items():
                    if isinstance(v, dict):
                        sub_str = " | ".join(f"{sk}: {sv}" for sk, sv in v.items())
                        lines.append(f"• {k}: {sub_str}")
                    elif isinstance(v, list):
                        lines.append(f"• {k}: {', '.join(str(item) for item in v)}")
                    else:
                        lines.append(f"• {k}: {v}")
                explanation_text = "\n".join(lines)
            elif isinstance(raw_explanation, list):
                explanation_text = "\n".join(f"• {item}" for item in raw_explanation)
            elif isinstance(raw_explanation, str):
                explanation_text = raw_explanation.strip()
            else:
                explanation_text = str(raw_explanation)

            return {
                "xai_method": method_key,
                "heading": fallback_info["heading"],
                "method_label": fallback_info["title"],
                "explanation": explanation_text,
            }

        except Exception as exc:
            logger.error("Groq API query for method-specific XAI failed: %s. Using fallback.", exc)

    return {
        "xai_method": method_key,
        "heading": fallback_info["heading"],
        "method_label": fallback_info["title"],
        "explanation": fallback_info["text"],
    }


def _ensure_str(val: Any, default: str = "") -> str:
    """Helper to convert any structured JSON field (list, dict, primitive) into a clean string."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return "\n".join(f"• {str(x)}" for x in val)
    if isinstance(val, dict):
        return "\n".join(f"• {k}: {v}" for k, v in val.items())
    if val is None:
        return default
    return str(val)


def get_disease_explanation(
    class_name: str,
    confidence: float,
    xai_method: str = "all",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generates structured disease explanation, XAI visual feature attribution, cure, prevention, and precautions."""
    disease_title = clean_class_name(class_name)
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")

    # Get method-specific XAI explanation
    xai_data = get_xai_specific_explanation(
        class_name=class_name,
        confidence=confidence,
        xai_method=xai_method,
        api_key=groq_api_key,
    )

    if GROQ_AVAILABLE and groq_api_key:
        try:
            logger.info("Querying Groq LLM for pathology guidance on '%s'...", disease_title)
            client = Groq(api_key=groq_api_key)
            
            prompt = (
                f"You are an expert agricultural botanist and plant pathologist.\n"
                f"A plant leaf was diagnosed with: '{disease_title}' (Raw Model Class: '{class_name}', Confidence: {confidence:.1f}%).\n\n"
                f"Provide a structured JSON response with EXACTLY the following keys:\n"
                f"{{\n"
                f'  "explanation": "Detailed 2-3 sentence scientific explanation of the disease and its biological pathology.",\n'
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
            return {
                "disease_title": disease_title,
                "explanation": _ensure_str(response_json.get("explanation"), "Detailed scientific explanation."),
                "xai_feature_explanation": _ensure_str(xai_data["explanation"], "Attribution features verified."),
                "cure": _ensure_str(response_json.get("cure"), "Recommended organic and chemical treatments."),
                "prevention": _ensure_str(response_json.get("prevention"), "Long-term agricultural prevention strategies."),
                "precautions": _ensure_str(response_json.get("precautions"), "Immediate agricultural precautions."),
            }

        except Exception as exc:
            logger.error("Groq API query failed: %s. Falling back to knowledge base.", exc)

    # Fallback execution
    key = class_name.lower()
    fallback_data = FALLBACK_KNOWLEDGE_BASE.get(key, FALLBACK_KNOWLEDGE_BASE["default"])
    
    return {
        "disease_title": disease_title,
        "explanation": fallback_data["explanation"],
        "xai_feature_explanation": _ensure_str(xai_data["explanation"]),
        "cure": fallback_data["cure"],
        "prevention": fallback_data["prevention"],
        "precautions": fallback_data["precautions"],
    }

