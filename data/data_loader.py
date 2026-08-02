"""DataLoader module for Explainable AI Plant Disease Detection.

Implements PyTorch Dataset and DataLoader creation for 70/15/15 split datasets.
"""

import json
import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from augmentation.augmentor import get_training_transforms
from config.config import TrainingConfig
from preprocessing.transforms import get_test_transforms, get_validation_transforms

logger = logging.getLogger("data_loader")


class PlantDiseaseDataset(Dataset):
    """Custom PyTorch Dataset for Plant Disease leaf images."""

    def __init__(
        self,
        split_dict: dict[str, list[str]],
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        """Initializes dataset from split manifest dictionary."""
        self.samples: list[tuple[str, int]] = []
        self.transform = transform

        for class_name, paths in split_dict.items():
            if class_name in class_to_idx:
                idx = class_to_idx[class_name]
                for p in paths:
                    self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img_tensor = self.transform(img)
                return img_tensor, label
        except Exception as exc:
            logger.error("Failed loading image file '%s': %s", image_path, exc)
            return torch.zeros(3, 224, 224), label


def create_data_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Creates PyTorch DataLoaders for Train (70%), Val (15%), and Test (15%) splits.

    Args:
        config: TrainingConfig instance.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names list).
    """
    with open(config.DATA_DIR / "class_mapping.json", "r", encoding="utf-8") as f:
        class_to_idx: dict[str, int] = json.load(f)

    with open(config.MODELS_DIR / config.CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    with open(config.DATA_DIR / "train_split.json", "r", encoding="utf-8") as f:
        train_split = json.load(f)

    with open(config.DATA_DIR / "val_split.json", "r", encoding="utf-8") as f:
        val_split = json.load(f)

    with open(config.DATA_DIR / "test_split.json", "r", encoding="utf-8") as f:
        test_split = json.load(f)

    train_dataset = PlantDiseaseDataset(train_split, class_to_idx, transform=get_training_transforms())
    val_dataset = PlantDiseaseDataset(val_split, class_to_idx, transform=get_validation_transforms())
    test_dataset = PlantDiseaseDataset(test_split, class_to_idx, transform=get_test_transforms())

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    logger.info(
        "DataLoaders Ready | Train: %d, Val: %d, Test: %d samples across %d classes.",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        len(class_names),
    )

    return train_loader, val_loader, test_loader, class_names
