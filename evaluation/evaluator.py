"""Evaluation module for Explainable AI Plant Disease Detection.

Generates evaluation_metrics.json, classification_report.json, confusion_matrix.png,
and per_class_accuracy.json inside reports/.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config.config import TrainingConfig
from utils.metrics import calculate_classification_metrics, calculate_per_class_accuracy

logger = logging.getLogger("evaluator")


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Runs comprehensive evaluation on test dataset and exports all reports.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        class_names: Sorted list of class names.
        device: Computing device.
        config: TrainingConfig instance.

    Returns:
        dict[str, Any]: Summary dictionary containing test metrics.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device)

    all_preds: list[int] = []
    all_targets: list[int] = []

    logger.info("Evaluating model on test dataset (%d samples)...", len(test_loader.dataset))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 1. Overall Metrics
    overall_metrics = calculate_classification_metrics(y_true, y_pred)
    logger.info(
        "Test Results | Accuracy: %.2f%% | Precision: %.2f%% | Recall: %.2f%% | F1 Score: %.2f%%",
        overall_metrics["accuracy"],
        overall_metrics["precision"],
        overall_metrics["recall"],
        overall_metrics["f1_score"],
    )

    metrics_path = config.REPORTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    # 2. Classification Report
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    clf_report_path = config.REPORTS_DIR / "classification_report.json"
    with open(clf_report_path, "w", encoding="utf-8") as f:
        json.dump(clf_report, f, indent=2)

    # 3. Per-Class Accuracy
    per_class_acc = calculate_per_class_accuracy(y_true, y_pred, class_names)
    per_class_path = config.REPORTS_DIR / "per_class_accuracy.json"
    with open(per_class_path, "w", encoding="utf-8") as f:
        json.dump(per_class_acc, f, indent=2)

    # 4. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Plant Disease Classification Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = config.REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    logger.info("Saved all evaluation reports inside '%s'.", config.REPORTS_DIR)
    return overall_metrics
