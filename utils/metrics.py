"""Metrics calculation utility for Explainable AI Plant Disease Detection.

Provides robust functions for calculating Accuracy, Precision, Recall, F1-score,
and per-class accuracy percentages.
"""

from typing import Any
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculates overall classification performance metrics.

    Args:
        y_true: True integer label array.
        y_pred: Predicted integer label array.

    Returns:
        dict[str, float]: Metrics dictionary containing accuracy, precision, recall, f1_score.
    """
    accuracy = float((y_true == y_pred).mean() * 100.0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": float(precision * 100.0),
        "recall": float(recall * 100.0),
        "f1_score": float(f1 * 100.0),
    }


def calculate_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Calculates per-class accuracy and sample counts.

    Args:
        y_true: Ground truth label array.
        y_pred: Predicted label array.
        class_names: List of class category names indexed by integer.

    Returns:
        dict[str, dict[str, Any]]: Mapping of class_name to total_samples, correct, and accuracy %.
    """
    per_class_results: dict[str, dict[str, Any]] = {}

    for idx, cls_name in enumerate(class_names):
        mask = (y_true == idx)
        total_samples = int(mask.sum())
        if total_samples > 0:
            correct = int((y_pred[mask] == idx).sum())
            acc = float((correct / total_samples) * 100.0)
        else:
            correct = 0
            acc = 0.0

        per_class_results[cls_name] = {
            "total_samples": total_samples,
            "correct_predictions": correct,
            "accuracy": round(acc, 2),
        }

    return per_class_results
