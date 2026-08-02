import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore

from explainability.grad_cam import GradCAM
from explainability.integrated_gradients import IntegratedGradients
from explainability.shap_explainer import SHAPExplainer
from models.efficientnet import build_efficientnet_b0


def test_grad_cam_heatmap_generation() -> None:
    """Verifies Grad-CAM heatmap output shape, type, and normalization range."""
    model = build_efficientnet_b0(num_classes=15, pretrained=False)
    target_layer = model.get_target_layer_for_gradcam()
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = grad_cam.generate_heatmap(input_tensor=dummy_input, target_class_idx=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_integrated_gradients_heatmap_generation() -> None:
    """Verifies Integrated Gradients heatmap output shape, type, and normalization range."""
    model = build_efficientnet_b0(num_classes=15, pretrained=False)
    ig_explainer = IntegratedGradients(model=model)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = ig_explainer.generate_heatmap(input_tensor=dummy_input, target_class_idx=0, steps=5)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_shap_heatmap_generation() -> None:
    """Verifies SHAP occlusion heatmap output shape, type, and normalization range."""
    model = build_efficientnet_b0(num_classes=15, pretrained=False)
    shap_explainer = SHAPExplainer(model=model, grid_size=4)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = shap_explainer.generate_heatmap(input_tensor=dummy_input, target_class_idx=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


if __name__ == "__main__":
    print("Running test_grad_cam_heatmap_generation...")
    test_grad_cam_heatmap_generation()
    print("Running test_integrated_gradients_heatmap_generation...")
    test_integrated_gradients_heatmap_generation()
    print("Running test_shap_heatmap_generation...")
    test_shap_heatmap_generation()
    print("All explainability unit tests passed successfully!")

