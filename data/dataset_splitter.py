"""Dataset splitter for Explainable AI Plant Disease Detection.

Auto-detects dataset location and class names, performs 70% Train, 15% Val, 15% Test
stratified split, and saves class_names.json inside models/ and manifests in data/.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from config.config import TrainingConfig

logger = logging.getLogger("dataset_splitter")

VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_classes(dataset_dir: Path) -> dict[str, list[Path]]:
    """Auto-detects dataset location and class categories.

    Args:
        dataset_dir: Root dataset directory.

    Returns:
        dict[str, list[Path]]: Class name mapped to list of image paths.
    """
    logger.info("Auto-detecting dataset location and classes in: %s", dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset location '{dataset_dir}' does not exist.")

    class_map: dict[str, list[Path]] = {}

    for root, _, files in os.walk(dataset_dir):
        images = [Path(root) / f for f in files if Path(f).suffix.lower() in VALID_EXTENSIONS]
        if images:
            rel_parts = Path(root).relative_to(dataset_dir).parts
            class_name = rel_parts[-1] if rel_parts else "unknown"

            if class_name not in class_map:
                class_map[class_name] = []
            class_map[class_name].extend(images)

    if not class_map:
        raise ValueError(f"No valid image files found in '{dataset_dir}'.")

    logger.info("Auto-detected %d distinct plant disease classes.", len(class_map))
    return class_map


def perform_stratified_split(
    class_map: dict[str, list[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Performs stratified split according to specified 70/15/15 ratios.

    Args:
        class_map: Dictionary mapping class names to image paths.
        train_ratio: Proportion of training data (default 0.70).
        val_ratio: Proportion of validation data (default 0.15).
        test_ratio: Proportion of test data (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        tuple containing train_dict, val_dict, test_dict, and sorted class_names list.
    """
    random.seed(seed)
    sorted_class_names = sorted(class_map.keys())

    train_data: dict[str, list[str]] = {c: [] for c in sorted_class_names}
    val_data: dict[str, list[str]] = {c: [] for c in sorted_class_names}
    test_data: dict[str, list[str]] = {c: [] for c in sorted_class_names}

    for c in sorted_class_names:
        paths = [str(p.resolve()) for p in class_map[c]]
        random.shuffle(paths)

        n_total = len(paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_data[c] = paths[:n_train]
        val_data[c] = paths[n_train : n_train + n_val]
        test_data[c] = paths[n_train + n_val :]

    return train_data, val_data, test_data, sorted_class_names


def prepare_dataset(config: TrainingConfig) -> list[str]:
    """Auto-detects dataset, splits data into 70/15/15, and saves class_names.json in models/.

    Args:
        config: Central TrainingConfig instance.

    Returns:
        list[str]: List of sorted class names.
    """
    config.create_directories()
    class_map = discover_classes(config.DATASET_DIR)

    train_dict, val_dict, test_dict, class_names = perform_stratified_split(
        class_map=class_map,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED,
    )

    # Save class_names.json in models/ directory as required
    class_names_path = config.MODELS_DIR / config.CLASS_NAMES_FILE
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Saved class_names.json inside '%s'.", class_names_path)

    # Also save class_mapping.json and split manifests in data/ directory
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    with open(config.DATA_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    with open(config.DATA_DIR / "train_split.json", "w", encoding="utf-8") as f:
        json.dump(train_dict, f, indent=2)

    with open(config.DATA_DIR / "val_split.json", "w", encoding="utf-8") as f:
        json.dump(val_dict, f, indent=2)

    with open(config.DATA_DIR / "test_split.json", "w", encoding="utf-8") as f:
        json.dump(test_dict, f, indent=2)

    train_count = sum(len(v) for v in train_dict.values())
    val_count = sum(len(v) for v in val_dict.values())
    test_count = sum(len(v) for v in test_dict.values())

    logger.info(
        "Split Completed (70/15/15): Train=%d, Val=%d, Test=%d across %d classes.",
        train_count,
        val_count,
        test_count,
        len(class_names),
    )

    return class_names
