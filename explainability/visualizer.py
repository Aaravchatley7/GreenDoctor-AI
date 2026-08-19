"""Visualizer utility for Explainable AI Plant Disease Detection.

Provides heatmap overlay, base64 encoding, and batch encoding utilities
for all XAI attribution methods: Grad-CAM, Grad-CAM++, Vanilla Saliency,
SmoothGrad, Integrated Gradients, SHAP, and LIME.
"""

import base64
import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("explainability.visualizer")


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """Overlays a normalized Grad-CAM heatmap onto the original image.

    Args:
        original_image: PIL Image of the leaf.
        heatmap: 2D float NumPy array in range [0, 1].
        alpha: Transparency blending factor for the heatmap (0.0 to 1.0).
        colormap: OpenCV colormap constant (default: COLORMAP_JET).

    Returns:
        Image.Image: Blended RGB PIL Image.
    """
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    orig_np = np.array(original_image)
    h, w, _ = orig_np.shape

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original image with color heatmap
    blended = np.float32(color_heatmap) * alpha + np.float32(orig_np) * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def encode_image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image into a base64 PNG data string.

    Args:
        image: PIL Image object.

    Returns:
        str: Base64 data URL string (data:image/png;base64,...).
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def encode_all_heatmaps(
    original_image: Image.Image,
    heatmaps: dict[str, np.ndarray | None],
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> dict[str, Any]:
    """Batch-overlays and base64-encodes a dictionary of named heatmaps.

    Reduces boilerplate in API routes by handling the overlay → encode pipeline
    for all XAI methods in a single call.

    Args:
        original_image: Original PIL leaf image.
        heatmaps: Mapping of XAI method name → 2D np.ndarray heatmap (or None to skip).
        alpha: Heatmap overlay transparency (0.0 = invisible, 1.0 = opaque).
        colormap: OpenCV colormap constant for heatmap colouring.

    Returns:
        dict[str, str | None]: Mapping of method name → base64 PNG data URL,
        or None if the corresponding heatmap was None.

    Example:
        >>> results = encode_all_heatmaps(
        ...     original_image=pil_img,
        ...     heatmaps={
        ...         "gradcam": gradcam_np,
        ...         "gradcam_pp": gradcam_pp_np,
        ...         "saliency": saliency_np,
        ...         "smoothgrad": smoothgrad_np,
        ...         "ig": ig_np,
        ...         "shap": shap_np,
        ...         "lime": None,  # skipped — not requested
        ...     },
        ... )
    """
    encoded: dict[str, Any] = {}
    for method_name, heatmap in heatmaps.items():
        if heatmap is None:
            encoded[method_name] = None
            continue
        try:
            overlay = overlay_heatmap_on_image(
                original_image=original_image,
                heatmap=heatmap,
                alpha=alpha,
                colormap=colormap,
            )
            encoded[method_name] = encode_image_to_base64(overlay)
        except Exception:
            logger.exception(
                "Failed to encode heatmap for method '%s'. Returning None.",
                method_name,
            )
            encoded[method_name] = None

    return encoded
