"""Dataset downloader module for Explainable AI Plant Disease Detection.

This module downloads the 'emmarex/plantdisease' dataset from Kaggle using
kagglehub, moves/copies the contents to the designated project DATASET folder,
and verifies dataset structure and file integrity.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import kagglehub
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_dataset")

DATASET_HANDLE: str = "emmarex/plantdisease"
DEFAULT_TARGET_DIR: Path = Path(__file__).resolve().parent.parent / "DATASET"


def download_kaggle_dataset(dataset_handle: str = DATASET_HANDLE) -> Path:
    """Downloads a dataset from Kaggle via kagglehub.

    Args:
        dataset_handle: Kaggle dataset handle (e.g. 'emmarex/plantdisease').

    Returns:
        Path: Local path where kagglehub cached the downloaded files.
    """
    logger.info("Initiating download for dataset handle: '%s'...", dataset_handle)
    downloaded_path_str: str = kagglehub.dataset_download(dataset_handle)
    downloaded_path: Path = Path(downloaded_path_str)
    logger.info("Successfully downloaded dataset to cache location: %s", downloaded_path)
    return downloaded_path


def copy_dataset_files(source_dir: Path, target_dir: Path) -> Path:
    """Copies downloaded dataset files into the target project directory.

    Args:
        source_dir: Source directory containing downloaded files.
        target_dir: Destination directory within the project folder.

    Returns:
        Path: Absolute path to the destination dataset folder.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Copying dataset files from '%s' to '%s'...", source_dir, target_dir)

    source_items = list(source_dir.iterdir())
    
    for item in source_items:
        dest_item = target_dir / item.name
        if item.is_dir():
            if dest_item.exists():
                logger.warning("Destination directory '%s' exists. Removing existing files...", dest_item)
                shutil.rmtree(dest_item)
            shutil.copytree(item, dest_item)
        else:
            shutil.copy2(item, dest_item)

    logger.info("Dataset successfully copied to target directory: %s", target_dir)
    return target_dir


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validates the contents and integrity of the downloaded dataset.

    Args:
        dataset_dir: Path to the dataset directory to validate.

    Returns:
        dict[str, Any]: Summary dictionary containing stats (folder count, total files, class counts).
    """
    logger.info("Starting dataset validation in directory: %s", dataset_dir)
    
    valid_extensions: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total_images: int = 0
    corrupted_images: int = 0
    class_stats: dict[str, int] = {}
    
    for root, dirs, files in os.walk(dataset_dir):
        image_files = [f for f in files if Path(f).suffix.lower() in valid_extensions]
        if image_files:
            rel_dir = Path(root).relative_to(dataset_dir)
            class_name = str(rel_dir)
            class_stats[class_name] = len(image_files)
            total_images += len(image_files)

            for img_name in image_files[:10]:
                img_path = Path(root) / img_name
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as exc:
                    logger.error("Corrupted image detected at %s: %s", img_path, exc)
                    corrupted_images += 1

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_dir),
        "total_categories": len(class_stats),
        "total_images": total_images,
        "corrupted_images": corrupted_images,
        "categories": class_stats,
    }

    logger.info(
        "Validation completed. Found %d sub-categories and %d total images (%d corrupted).",
        len(class_stats),
        total_images,
        corrupted_images,
    )
    return summary


def main() -> None:
    """Main execution entry point for downloading and validating the dataset."""
    logger.info("Starting dataset acquisition workflow...")
    cache_path: Path = download_kaggle_dataset(DATASET_HANDLE)
    dataset_path: Path = copy_dataset_files(cache_path, DEFAULT_TARGET_DIR)
    summary: dict[str, Any] = validate_dataset(dataset_path)
    
    logger.info("=== DATASET ACQUISITION SUMMARY ===")
    logger.info("Dataset Path: %s", summary["dataset_path"])
    logger.info("Total Categories: %d", summary["total_categories"])
    logger.info("Total Images: %d", summary["total_images"])
    logger.info("Corrupted Images: %d", summary["corrupted_images"])
    for cat, count in list(summary["categories"].items())[:15]:
        logger.info(" - %s: %d images", cat, count)
    if len(summary["categories"]) > 15:
        logger.info(" ... and %d more categories.", len(summary["categories"]) - 15)


if __name__ == "__main__":
    main()
