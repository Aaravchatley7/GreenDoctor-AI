"""Visualizer utility for Explainable AI Plant Disease Detection.

Overlays Grad-CAM heatmaps onto original leaf images and encodes output visualizations.
"""

import base64
import io
import logging

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
