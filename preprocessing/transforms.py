"""Preprocessing module for Explainable AI Plant Disease Detection.

Implements deterministic preprocessing (resizing to 224x224 and ImageNet normalization)
for Validation, Test, and Inference.
"""

import torch
from PIL import Image
from torchvision import transforms
from config.config import TrainingConfig

config = TrainingConfig()


def get_validation_transforms() -> transforms.Compose:
    """Returns deterministic transforms for validation and test evaluation.

    Includes 224x224 resize and ImageNet normalization without any data augmentation.

    Returns:
        transforms.Compose: Validation image transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_test_transforms() -> transforms.Compose:
    """Returns deterministic transforms for test evaluation (identical to validation).

    Returns:
        transforms.Compose: Test image transformation pipeline.
    """
    return get_validation_transforms()


def preprocess_image_for_inference(image: Image.Image) -> torch.Tensor:
    """Preprocesses a single PIL Image into a normalized tensor ready for model inference.

    Args:
        image: PIL Image object.

    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 3, 224, 224).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    transform = get_validation_transforms()
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension
