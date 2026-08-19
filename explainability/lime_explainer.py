"""LIME Explainable AI Engine for Plant Disease Detection.

Implements Local Interpretable Model-agnostic Explanations (LIME) for image
classification using superpixel-based perturbation analysis.

LIME (Ribeiro et al., 2016) treats the model as a complete black box and explains
individual predictions by:
    1. Segmenting the input image into semantically coherent superpixels (SLIC algorithm).
    2. Generating a neighbourhood of perturbed images where random subsets of
       superpixels are occluded (set to grey).
    3. Getting model predictions on all perturbed images.
    4. Fitting a weighted linear surrogate model (LASSO/Ridge) around these predictions.
    5. The linear model's coefficients are the SHAP-like feature importances per superpixel.

LIME is the only truly model-agnostic method in our XAI stack — it requires no access
to gradients, layers, or internal model structure. It is particularly valuable for
non-ML audiences because the "which regions matter" question is answered via simple
linear approximation, not deep network internals.

Trade-off: LIME is the slowest method (~2–5 seconds) due to the ~150–200 forward passes
required to build the perturbation dataset.

Reference:
    Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
    Explaining the predictions of any classifier. KDD 2016. arXiv:1602.04938.
"""

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger("explainability.lime_explainer")


class LIMEExplainer:
    """LIME superpixel explainer for plant disease classification.

    Generates superpixel importance maps by probing the model with hundreds of
    masked perturbations and fitting a linear surrogate model. Compatible with
    the overlay_heatmap_on_image() visualizer interface.
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 200,
        num_features: int = 10,
    ) -> None:
        """Initializes the LIME explainer.

        Args:
            model: Trained PyTorch classification model (EfficientNet-B0).
            num_samples: Number of perturbed images to generate for the surrogate
                         model fit (default: 200). Higher → more stable, slower.
            num_features: Number of top superpixels to highlight in the output
                          (default: 10). Controls how many regions are shown as
                          important.
        """
        self.model = model
        self.num_samples = num_samples
        self.num_features = num_features
        logger.info(
            "LIME explainer initialized. num_samples=%d, num_features=%d",
            num_samples,
            num_features,
        )

    def _build_predict_fn(
        self,
        device: torch.device,
        preprocess_fn: Callable[[Image.Image], torch.Tensor],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Builds a LIME-compatible prediction function.

        LIME's LimeImageExplainer expects a function that takes an array of
        perturbed uint8 images (N, H, W, 3) and returns a probability matrix (N, C).

        Args:
            device: Torch device for inference.
            preprocess_fn: Preprocessing transform function for a single PIL Image.

        Returns:
            Callable: Batch prediction function compatible with lime.lime_image.
        """
        model = self.model

        def predict_batch(images: np.ndarray) -> np.ndarray:
            """Runs batch inference on perturbed images for LIME surrogate fitting.

            Args:
                images: NumPy array of perturbed uint8 leaf images (N, H, W, 3).

            Returns:
                np.ndarray: Softmax probability matrix (N, C).
            """
            model.eval()
            batch_tensors = []
            for img_array in images:
                pil_img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
                tensor = preprocess_fn(pil_img)        # (1, 3, H, W)
                batch_tensors.append(tensor)

            batch = torch.cat(batch_tensors, dim=0).to(device)  # (N, 3, H, W)

            with torch.no_grad():
                logits = model(batch)                  # (N, C)
                probs = torch.softmax(logits, dim=1)   # (N, C)

            return probs.cpu().numpy()

        return predict_batch

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int | None = None,
        original_image: Image.Image | None = None,
    ) -> np.ndarray:
        """Generates a LIME superpixel importance heatmap normalized to [0, 1].

        This method requires the `lime` package (pip install lime).
        It runs ~200 forward passes through the model and returns a spatial
        map of which superpixels positively influenced the target class prediction.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W) — used to infer
                          the target class and get the device.
            target_class_idx: Class index to explain. Defaults to argmax prediction.
            original_image: Original PIL Image (before preprocessing) to run LIME on.
                            LIME works on raw pixel space, not the normalized tensor.
                            If None, the tensor is denormalized and converted back.

        Returns:
            np.ndarray: Normalized 2D superpixel importance heatmap (H, W), [0, 1].

        Raises:
            ImportError: If the `lime` package is not installed.
            RuntimeError: If the LIME explanation fails or returns no segments.
        """
        try:
            from lime import lime_image
            from lime.wrappers.scikit_image import SegmentationAlgorithm
        except ImportError as exc:
            raise ImportError(
                "LIME package is required. Install with: pip install lime"
            ) from exc

        from preprocessing.transforms import preprocess_image_for_inference

        self.model.eval()
        device = next(self.model.parameters()).device

        # Determine target class
        with torch.no_grad():
            logits = self.model(input_tensor)
            if target_class_idx is None:
                target_class_idx = int(logits.argmax(dim=1).item())

        # Reconstruct original PIL image from tensor if not explicitly provided
        if original_image is None:
            # De-normalize the tensor (ImageNet mean/std) for LIME
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = input_tensor.squeeze(0).cpu() * imagenet_std + imagenet_mean
            img_denorm = torch.clamp(img_denorm, 0.0, 1.0)
            img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            original_image = Image.fromarray(img_np, mode="RGB")

        img_array = np.array(original_image)

        # Build LIME-compatible prediction function
        predict_fn = self._build_predict_fn(
            device=device,
            preprocess_fn=preprocess_image_for_inference,
        )

        # Initialize LIME image explainer with SLIC superpixel segmentation
        explainer = lime_image.LimeImageExplainer(verbose=False)
        segmentation_fn = SegmentationAlgorithm(
            algo_type="slic",
            n_segments=150,
            compactness=10,
            sigma=1,
        )

        logger.info(
            "Running LIME with %d samples for class %d...",
            self.num_samples,
            target_class_idx,
        )

        explanation = explainer.explain_instance(
            image=img_array,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=128,           # Grey occlusion color for masked superpixels
            num_samples=self.num_samples,
            segmentation_fn=segmentation_fn,
        )

        # Extract importance mask for top positive superpixels
        # get_image_and_mask returns (image, mask) where mask has 1 for important regions
        _, mask = explanation.get_image_and_mask(
            label=target_class_idx,
            positive_only=True,
            num_features=self.num_features,
            hide_rest=False,
        )

        # Build a continuous importance heatmap from LIME segment weights
        # Access the underlying segment importance scores
        segment_weights: dict[int, float] = dict(
            explanation.local_exp[target_class_idx]
        )
        segments = explanation.segments   # (H, W) integer segment ID map

        heatmap = np.zeros_like(segments, dtype=np.float32)
        for seg_id, weight in segment_weights.items():
            # Only encode positive-influence superpixels (those helping the prediction)
            if weight > 0.0:
                heatmap[segments == seg_id] = weight

        # Normalize to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = mask.astype(np.float32)   # Fall back to binary mask

        logger.info(
            "LIME heatmap generated. Active segments: %d, Shape: %s",
            int((heatmap > 0).sum()),
            heatmap.shape,
        )
        return heatmap
