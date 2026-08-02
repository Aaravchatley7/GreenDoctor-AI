"""Grad-CAM Explainable AI Engine for Plant Disease Detection.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNet-B0
to highlight regions of interest influencing model predictions.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("explainability.grad_cam")


class GradCAM:
    """Grad-CAM explainer class for PyTorch Convolutional Neural Networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initializes Grad-CAM with a target model and target feature layer.

        Args:
            model: Trained PyTorch classification model.
            target_layer: The target convolutional layer (e.g. model.base_model.features[8]).
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        """Hook to capture layer forward activation feature maps."""
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Hook to capture layer backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int | None = None,
    ) -> np.ndarray:
        """Generates a Grad-CAM heatmap array normalized to range [0, 1].

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class_idx: Index of the target class to explain. Defaults to top predicted class.

        Returns:
            np.ndarray: Normalized 2D heatmap matrix of shape (H, W).
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, target_class_idx]

        # Zero existing gradients and backward target score
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations during Grad-CAM execution.")

        # Compute importance weights via Global Average Pooling of gradients
        gradients = self.gradients[0]  # shape: (C, H, W)
        activations = self.activations[0]  # shape: (C, H, W)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # shape: (C, 1, 1)

        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=0)  # shape: (H, W)

        # Apply ReLU activation to keep only positive influence regions
        cam = F.relu(cam)

        # Resize heatmap to match original input tensor height & width
        cam = cam.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        # Normalize heatmap to [0, 1] range
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()

        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap
