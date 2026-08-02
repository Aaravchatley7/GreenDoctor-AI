"""Integrated Gradients Explainable AI Engine for Plant Disease Detection.

Implements Integrated Gradients attribution method for PyTorch classification models
to attribute model predictions to input pixel intensities.

Uses per-step autograd.grad calls for full MPS/CUDA/CPU backend compatibility.
"""

import logging
from typing import Optional

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore

logger = logging.getLogger("explainability.integrated_gradients")


class IntegratedGradients:
    """Integrated Gradients explainer class for PyTorch neural network models.

    Computes per-step gradients via torch.autograd.grad — compatible with
    MPS (Apple Silicon), CUDA, and CPU backends.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initializes Integrated Gradients with a trained PyTorch model.

        Args:
            model: Trained PyTorch classification model.
        """
        self.model = model

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        steps: int = 20,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generates an Integrated Gradients heatmap array normalized to range [0, 1].

        Computes gradients at each interpolation step independently using
        torch.autograd.grad, which is fully compatible with MPS, CUDA, and CPU.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class_idx: Index of target class. Defaults to top predicted class.
            steps: Number of linear interpolation steps between baseline and input.
            baseline: Baseline reference tensor (default: zero tensor matching input shape).

        Returns:
            np.ndarray: Normalized 2D heatmap matrix of shape (H, W).
        """
        self.model.eval()
        device = input_tensor.device

        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(device)

        # Detach for clean computation graph
        input_tensor = input_tensor.detach()
        baseline = baseline.detach()

        # 1. Determine target class index if not provided
        if target_class_idx is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                target_class_idx = int(logits.argmax(dim=1).item())

        # 2. Accumulate per-step gradients via autograd.grad
        grads_list: list[torch.Tensor] = []

        for i in range(steps + 1):
            alpha = float(i) / steps
            # Create a fresh leaf tensor for each step — required for autograd.grad on MPS
            step_input = (baseline + alpha * (input_tensor - baseline)).clone()
            step_input.requires_grad_(True)

            logits = self.model(step_input)
            score = logits[0, target_class_idx]

            # autograd.grad works reliably on all backends (MPS, CUDA, CPU)
            grad = torch.autograd.grad(
                outputs=score,
                inputs=step_input,
                create_graph=False,
                retain_graph=False,
            )[0]  # Shape: (1, 3, H, W)

            grads_list.append(grad.detach())

        # 3. Stack into (steps+1, 1, 3, H, W) then mean across steps
        grads_tensor = torch.stack(grads_list, dim=0)  # (steps+1, 1, 3, H, W)
        avg_grads = grads_tensor.mean(dim=0)  # (1, 3, H, W)

        # 4. Element-wise product with (input - baseline) → Integrated Gradients
        integrated_grad = (input_tensor - baseline) * avg_grads  # (1, 3, H, W)

        # 5. Aggregate across color channels & compute absolute intensity
        attr = torch.sum(torch.abs(integrated_grad[0]), dim=0)  # (H, W)

        # 6. Normalize to [0, 1] range
        attr_np = attr.cpu().numpy()
        attr_min, attr_max = attr_np.min(), attr_np.max()

        if attr_max - attr_min > 1e-8:
            heatmap = (attr_np - attr_min) / (attr_max - attr_min)
        else:
            heatmap = np.zeros_like(attr_np)

        logger.info("Integrated Gradients heatmap generated successfully (%d steps).", steps)
        return heatmap
