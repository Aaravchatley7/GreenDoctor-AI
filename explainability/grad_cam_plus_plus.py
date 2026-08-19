"""Grad-CAM++ Explainable AI Engine for Plant Disease Detection.

Implements Gradient-weighted Class Activation Mapping++ (Grad-CAM++) introduced by
Chattopadhay et al. (2018). Grad-CAM++ improves upon Grad-CAM by using a pixel-wise
weighting of gradients (via second-order partial derivatives) rather than a global
average pool, producing sharper and more accurate localization — especially when
multiple disease instances appear in the same leaf image.

Reference:
    Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018).
    Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional
    networks. In WACV 2018.

Interface:
    Identical to GradCAM — generate_heatmap(input_tensor, target_class_idx) → np.ndarray
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("explainability.grad_cam_plus_plus")


class GradCAMPlusPlus:
    """Grad-CAM++ explainer for PyTorch Convolutional Neural Networks.

    Produces sharper class-discriminative heatmaps compared to standard Grad-CAM
    by computing pixel-wise second-order gradient importance weights.
    Particularly effective for multi-instance disease localization on a single leaf.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initializes Grad-CAM++ with a target model and convolutional layer.

        Args:
            model: Trained PyTorch classification model (EfficientNet-B0).
            target_layer: Target convolutional layer, e.g. model.base_model.features[8].
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

        logger.info(
            "Grad-CAM++ engine initialized on layer: %s",
            type(target_layer).__name__,
        )

    def _forward_hook(
        self,
        module: nn.Module,
        input: tuple,  # noqa: A002
        output: torch.Tensor,
    ) -> None:
        """Captures forward-pass activation feature maps."""
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Captures backward-pass gradient tensors."""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int | None = None,
    ) -> np.ndarray:
        """Generates a Grad-CAM++ heatmap normalized to [0, 1].

        The key difference from Grad-CAM: instead of global-average-pooling the
        gradients into scalar channel weights (alpha_k), Grad-CAM++ computes
        pixel-wise alpha weights using the second-order gradient formula:

            alpha_k_ij = (d²y^c / dA^k_ij²) /
                         (2 * d²y^c / dA^k_ij² + sum_ab(A^k_ab * d³y^c / dA^k_ij³))

        In practice, this is approximated via the ReLU-gated gradient:
            alpha_k = ReLU(grad)² / (2*ReLU(grad)² + sum(A * ReLU(grad)³) + eps)

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class_idx: Target class index to explain. Defaults to argmax.

        Returns:
            np.ndarray: Normalized 2D heatmap of shape (H, W), values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass — activations captured via hook
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, target_class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM++ hooks did not capture gradients or activations."
            )

        # Shapes: (C, h, w)
        grads = self.gradients[0]       # dY/dA
        acts  = self.activations[0]     # A^k feature maps

        # --- Grad-CAM++ alpha weight computation ---
        # Apply ReLU to gradients — only positive gradients contribute
        grads_relu = F.relu(grads)

        # Numerator: (relu_grad)²
        numerator = grads_relu ** 2

        # Denominator: 2*(relu_grad)² + sum over spatial dims(A * (relu_grad)³)
        sum_acts = acts * (grads_relu ** 3)
        sum_acts = sum_acts.sum(dim=(1, 2), keepdim=True)  # global spatial sum per channel
        denominator = 2.0 * numerator + sum_acts + 1e-8

        # Alpha weights: pixel-wise, then globally averaged per channel
        alpha = numerator / denominator                          # (C, h, w)
        alpha_weights = (alpha * F.relu(grads)).mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.einsum("c,chw->hw", alpha_weights, acts)    # (h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)                     # (1, 1, h, w)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        # Min-max normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()

        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        logger.debug(
            "Grad-CAM++ heatmap generated. Shape: %s, Max: %.4f",
            heatmap.shape,
            heatmap.max(),
        )
        return heatmap
