"""Saliency Map Explainable AI Engine for Plant Disease Detection.

Implements two gradient-based pixel-level attribution techniques:

1. **Vanilla Saliency Map** (Simonyan et al., 2013):
   Computes the gradient of the predicted class score with respect to each input pixel
   (∂y^c / ∂x). Pixels with high absolute gradient magnitude most strongly influence
   the model's output when perturbed. The fastest pixel-level XAI method (<3ms).

2. **SmoothGrad** (Smilkov et al., 2017):
   Addresses the noisiness of vanilla saliency by averaging gradients computed over
   N=25 slightly noisy versions of the input image (Gaussian noise, σ=0.1). The
   resulting map is significantly cleaner and more visually interpretable.

Both return a normalized 2D heatmap compatible with overlay_heatmap_on_image().

References:
    - Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional
      networks: Visualising image classification models and saliency maps. arXiv:1312.6034.
    - Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017).
      SmoothGrad: removing noise by adding noise. arXiv:1706.03825.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("explainability.saliency")


class SaliencyExplainer:
    """Vanilla gradient saliency map explainer for EfficientNet-B0.

    Computes ∂(class_score) / ∂(input_pixel) in a single backward pass.
    The absolute gradient magnitude map is used as pixel importance.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initializes the Saliency explainer with a trained model.

        Args:
            model: Trained PyTorch classification model (EfficientNet-B0).
        """
        self.model = model
        logger.info("Saliency Map explainer initialized.")

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int | None = None,
    ) -> np.ndarray:
        """Generates a vanilla saliency heatmap via a single backward pass.

        For each input pixel x_i, the saliency S_i = |∂y^c / ∂x_i|.
        Channels are aggregated by max across the 3 RGB channels to produce
        a single 2D importance map.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class_idx: Target class index. Defaults to argmax prediction.

        Returns:
            np.ndarray: Normalized saliency heatmap of shape (H, W), values in [0, 1].
        """
        self.model.eval()

        # Clone and enable gradient tracking on the input
        inp = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        logits = self.model(inp)

        if target_class_idx is None:
            target_class_idx = int(logits.argmax(dim=1).item())

        # Backward on the target class score
        self.model.zero_grad()
        score = logits[0, target_class_idx]
        score.backward()

        if inp.grad is None:
            raise RuntimeError("Input gradient is None — backward pass did not populate gradients.")

        # Aggregate across RGB channels: take max absolute value per pixel
        # Shape: (1, 3, H, W) → (H, W)
        saliency = inp.grad.data.abs()
        saliency, _ = torch.max(saliency, dim=1)   # (1, H, W)
        saliency = saliency.squeeze(0)              # (H, W)

        # Normalize to [0, 1]
        sal_np = saliency.cpu().numpy()
        s_min, s_max = sal_np.min(), sal_np.max()

        if s_max - s_min > 1e-8:
            heatmap = (sal_np - s_min) / (s_max - s_min)
        else:
            heatmap = np.zeros_like(sal_np)

        logger.debug(
            "Saliency heatmap generated. Shape: %s, Max: %.4f",
            heatmap.shape,
            heatmap.max(),
        )
        return heatmap


class SmoothGradExplainer:
    """SmoothGrad explainer — noise-averaged saliency for cleaner attributions.

    Reduces the visual noise inherent in vanilla saliency maps by averaging
    gradient maps computed over N perturbed copies of the input image.
    Each copy has Gaussian noise N(0, σ) added. The averaged map reveals
    stable, semantically meaningful features.
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 25,
        noise_level: float = 0.1,
    ) -> None:
        """Initializes SmoothGrad with averaging parameters.

        Args:
            model: Trained PyTorch classification model.
            num_samples: Number of noisy samples to average over (default: 25).
            noise_level: Standard deviation of Gaussian noise as a fraction of
                         the input value range (default: 0.1 → σ = 0.1 * (max - min)).
        """
        self.model = model
        self.num_samples = num_samples
        self.noise_level = noise_level
        logger.info(
            "SmoothGrad explainer initialized. Samples=%d, Noise=%.2f",
            num_samples,
            noise_level,
        )

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int | None = None,
    ) -> np.ndarray:
        """Generates a SmoothGrad heatmap by averaging N noisy saliency passes.

        Algorithm:
            σ = noise_level * (max(x) - min(x))
            For i in 1..N:
                x̃ = x + N(0, σ)
                Compute saliency(x̃, c)
            SmoothGrad = (1/N) * Σ saliency(x̃_i, c)

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class_idx: Target class index. Defaults to argmax prediction.

        Returns:
            np.ndarray: Normalized SmoothGrad heatmap of shape (H, W), values in [0, 1].
        """
        self.model.eval()

        # Determine target class from clean forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            if target_class_idx is None:
                target_class_idx = int(logits.argmax(dim=1).item())

        # Compute noise standard deviation based on input range
        x_min = input_tensor.min().item()
        x_max = input_tensor.max().item()
        sigma = self.noise_level * (x_max - x_min)

        accumulated_gradients = torch.zeros_like(input_tensor)

        for _ in range(self.num_samples):
            # Generate a noisy copy of the input
            noise = torch.randn_like(input_tensor) * sigma
            noisy_input = (input_tensor + noise).detach().requires_grad_(True)

            # Forward pass on noisy input
            logits_noisy = self.model(noisy_input)
            self.model.zero_grad()

            # Backward on target class score
            score = logits_noisy[0, target_class_idx]
            score.backward()

            if noisy_input.grad is not None:
                accumulated_gradients += noisy_input.grad.data.abs()

        # Average gradients across all samples
        avg_grads = accumulated_gradients / self.num_samples  # (1, 3, H, W)

        # Aggregate across RGB channels via max
        smooth_map, _ = torch.max(avg_grads, dim=1)  # (1, H, W)
        smooth_map = smooth_map.squeeze(0)            # (H, W)

        # Normalize to [0, 1]
        sm_np = smooth_map.cpu().numpy()
        sm_min, sm_max = sm_np.min(), sm_np.max()

        if sm_max - sm_min > 1e-8:
            heatmap = (sm_np - sm_min) / (sm_max - sm_min)
        else:
            heatmap = np.zeros_like(sm_np)

        logger.debug(
            "SmoothGrad heatmap generated. Samples=%d, Shape=%s, Max=%.4f",
            self.num_samples,
            heatmap.shape,
            heatmap.max(),
        )
        return heatmap
