"""Master training pipeline entry point for Explainable AI Plant Disease Detection.

Orchestrates 70/15/15 dataset splitting, two-stage transfer learning training,
checkpoint saving, and test evaluation report generation.
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config.config import TrainingConfig
from data.data_loader import create_data_loaders
from data.dataset_splitter import prepare_dataset
from evaluation.evaluator import evaluate_test_set
from models.efficientnet import build_efficientnet_b0
from training.trainer import TwoStageTrainer
from utils.logger import setup_logger


def main() -> None:
    """Runs the complete end-to-end training and evaluation pipeline."""
    config = TrainingConfig()
    config.create_directories()

    logger = setup_logger(config.LOGS_DIR, name="train_main")
    logger.info("==================================================================")
    logger.info("  EXPLAINABLE AI PLANT DISEASE DETECTION - TWO-STAGE PIPELINE     ")
    logger.info("==================================================================")

    try:
        # 1. Dataset Preparation & 70/15/15 Stratified Split
        logger.info("Step 1: Preparing dataset and 70/15/15 stratified split...")
        class_names = prepare_dataset(config)

        # 2. DataLoaders Setup
        logger.info("Step 2: Creating PyTorch DataLoaders...")
        train_loader, val_loader, test_loader, class_names = create_data_loaders(config)

        # 3. Model Instantiation
        logger.info("Step 3: Building EfficientNet-B0 model architecture for %d classes...", len(class_names))
        model = build_efficientnet_b0(num_classes=len(class_names), pretrained=config.PRETRAINED)

        # 4. Two-Stage Transfer Learning Training
        logger.info("Step 4: Starting Two-Stage Transfer Learning...")
        trainer = TwoStageTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        training_summary = trainer.run_training()

        # 5. Final Evaluation on Test Set using Best Checkpoint
        logger.info("Step 5: Evaluating Best Model Checkpoint on Test Dataset...")
        best_checkpoint_path = config.MODELS_DIR / config.BEST_MODEL_NAME
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=trainer.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Successfully loaded best model checkpoint from '%s'.", best_checkpoint_path)

        test_metrics = evaluate_test_set(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            device=trainer.device,
            config=config,
        )

        # Calculate model file size
        model_size_mb = os.path.getsize(best_checkpoint_path) / (1024 * 1024) if best_checkpoint_path.exists() else 0.0

        # Print Final Summary Box
        print("\n" + "=" * 65)
        print("          TRAINING PIPELINE EXECUTION COMPLETE SUMMARY           ")
        print("=" * 65)
        print(f" 💻 Selected Device           : {trainer.device_name}")
        print(f" 🔄 Total Epochs Trained     : {training_summary['total_epochs']} / {config.MAX_TOTAL_EPOCHS}")
        print(f" 🎯 Best Validation Accuracy : {training_summary['best_val_acc']:.2f}%")
        print(f" 📈 Best Validation Precision: {training_summary['best_precision']:.2f}%")
        print(f" 🔄 Best Validation Recall   : {training_summary['best_recall']:.2f}%")
        print(f" 📊 Best Validation F1 Score : {training_summary['best_f1_score']:.2f}%")
        print(f" ⏱️ Total Training Time     : {time.strftime('%H:%M:%S', time.gmtime(training_summary['total_time_seconds']))}")
        print(f" 🛑 Early Stopping Triggered : {'Yes' if training_summary['early_stopped'] else 'No'}")
        print(f" 💾 Best Checkpoint Path    : {best_checkpoint_path}")
        print(f" 📦 Model File Size          : {model_size_mb:.2f} MB")
        print("=" * 65)

        logger.info("==================================================================")
        logger.info("  PIPELINE FINISHED SUCCESSFULLY                                  ")
        logger.info("  - Test Accuracy: %.2f%%", test_metrics["accuracy"])
        logger.info("  - Test F1 Score: %.2f%%", test_metrics["f1_score"])
        logger.info("  - All Reports Saved to: %s", config.REPORTS_DIR)
        logger.info("==================================================================")

    except Exception as exc:
        logger.error("Pipeline execution encountered an error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
