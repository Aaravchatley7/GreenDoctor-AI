"""CLI Evaluation Script for Explainable AI Plant Disease Detection.

Evaluates the best trained model on the test dataset and prints Accuracy,
Precision, Recall, F1-Score, Classification Report, and Confusion Matrix location.

Usage:
    .venv/bin/python evaluate.py
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config.config import TrainingConfig
from data.data_loader import create_data_loaders
from evaluation.evaluator import evaluate_test_set
from models.efficientnet import build_efficientnet_b0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def run_evaluation() -> None:
    """Loads best model checkpoint and evaluates test dataset metrics."""
    config = TrainingConfig()
    checkpoint_path = config.MODELS_DIR / config.BEST_MODEL_NAME
    class_names_path = config.MODELS_DIR / config.CLASS_NAMES_FILE

    if not checkpoint_path.exists():
        logger.error("Best model checkpoint not found at '%s'. Run training first.", checkpoint_path)
        sys.exit(1)

    if not class_names_path.exists():
        logger.error("Class names file not found at '%s'. Run training first.", class_names_path)
        sys.exit(1)

    # 1. Load Class Names & DataLoaders
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    _, _, test_loader, _ = create_data_loaders(config)

    # 2. Select Device & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Running evaluation on device: %s (%d test samples)...", device, len(test_loader.dataset))

    model = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 3. Evaluate Metrics & Save Reports
    metrics = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        config=config,
    )

    # 4. Read generated classification report
    clf_report_path = config.REPORTS_DIR / "classification_report.json"
    with open(clf_report_path, "r", encoding="utf-8") as f:
        clf_report = json.load(f)

    # 5. Display Formatted Results in Terminal
    print("\n" + "=" * 70)
    print("        MODEL EVALUATION REPORT — PLANT DISEASE DETECTION         ")
    print("=" * 70)
    print(f" 🎯 Test Accuracy          : {metrics['accuracy']:.2f}%")
    print(f" 📈 Weighted Precision     : {metrics['precision']:.2f}%")
    print(f" 🔄 Weighted Recall        : {metrics['recall']:.2f}%")
    print(f" 📊 Weighted F1-Score      : {metrics['f1_score']:.2f}%")
    print(f" 🖼️ Confusion Matrix Plot   : {config.REPORTS_DIR / 'confusion_matrix.png'}")
    print(f" 📄 Full Report JSON       : {config.REPORTS_DIR / 'classification_report.json'}")
    print("-" * 70)
    print(" PER-CLASS METRICS BREAKDOWN:")
    print("-" * 70)
    print(f" {'Category Name':<42} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 70)

    for cls_name in class_names:
        if cls_name in clf_report:
            p = clf_report[cls_name]["precision"] * 100.0
            r = clf_report[cls_name]["recall"] * 100.0
            f1 = clf_report[cls_name]["f1-score"] * 100.0
            print(f" {cls_name:<42} | {p:>8.2f}% | {r:>6.2f}% | {f1:>6.2f}%")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()
