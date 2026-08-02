"""Model architecture module for Explainable AI Plant Disease Detection.

Implements two-stage transfer learning with EfficientNet-B0 while preserving Grad-CAM hooks.
"""

import logging
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger("models.efficientnet")


class EfficientNetB0PlantClassifier(nn.Module):
    """EfficientNet-B0 plant disease classifier supporting two-stage transfer learning."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Initializes the EfficientNet-B0 model with custom classifier head.

        Args:
            num_classes: Number of disease classes to classify.
            pretrained: Whether to load ImageNet pre-trained weights.
        """
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.base_model = efficientnet_b0(weights=weights)

        in_features: int = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )
        logger.info("Initialized EfficientNet-B0 for %d classes.", num_classes)

    def freeze_backbone(self) -> None:
        """Stage 1: Freezes the entire feature extraction backbone."""
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        # Ensure classifier head is trainable
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
        logger.info("Stage 1: EfficientNet-B0 backbone frozen. Only classifier head is trainable.")

    def unfreeze_final_blocks(self, unfreeze_from_block: int = 6) -> None:
        """Stage 2: Unfreezes the final EfficientNet blocks for fine-tuning.

        Args:
            unfreeze_from_block: Index from which feature blocks will be unfrozen (0 to 8).
        """
        for i, block in enumerate(self.base_model.features):
            requires_grad = i >= unfreeze_from_block
            for param in block.parameters():
                param.requires_grad = requires_grad

        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

        logger.info("Stage 2: Unfrozen EfficientNet-B0 blocks from index %d onwards for fine-tuning.", unfreeze_from_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.base_model(x)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the target layer for future Grad-CAM integration without modifying training logic.

        Returns:
            nn.Module: The final feature map extraction layer (features[8]).
        """
        return self.base_model.features[8]


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> EfficientNetB0PlantClassifier:
    """Factory function creating an EfficientNetB0PlantClassifier instance.

    Args:
        num_classes: Number of target classes.
        pretrained: Whether to load pre-trained weights.

    Returns:
        EfficientNetB0PlantClassifier: PyTorch model instance.
    """
    return EfficientNetB0PlantClassifier(num_classes=num_classes, pretrained=pretrained)
