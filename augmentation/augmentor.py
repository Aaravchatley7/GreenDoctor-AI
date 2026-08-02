"""Augmentation module for Explainable AI Plant Disease Detection.

Implements training image augmentations: Random Horizontal Flip, Random Rotation,
Random Brightness & Contrast, and Random Zoom.
"""

from torchvision import transforms
from config.config import TrainingConfig

config = TrainingConfig()


def get_training_transforms() -> transforms.Compose:
    """Returns training data augmentation pipeline.

    Includes:
    - Random Zoom (RandomResizedCrop with scale 0.8 to 1.0)
    - Random Horizontal Flip (p=0.5)
    - Random Rotation (up to 20 degrees)
    - Random Brightness & Contrast (ColorJitter)
    - ToTensor & ImageNet Normalization

    Returns:
        transforms.Compose: Training augmentation pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
