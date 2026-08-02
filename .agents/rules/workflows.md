---
trigger: always_on
---

Goal:
Train a plant disease detection model.

Steps:

1. Verify dataset.
2. Split train validation test.
3. Apply augmentation.
4. Train model.
5. Save best checkpoint.
6. Evaluate metrics.
7. Generate confusion matrix.
8. Save training report.

🌿 Workflow 2 — Add New Disease

Goal:
Add support for a new disease.

Steps:

Update dataset.

Update class labels.

Retrain model.

Evaluate.

Generate Grad-CAM.

Update documentation.


🔥 Workflow 3 — Explain Prediction


Goal:
Explain a prediction.

Steps:

Run inference.

Generate confidence.

Generate Grad-CAM.

Overlay heatmap.

Explain which regions influenced the prediction.

Export visualization.



📊 Workflow 4 — Final Evaluation
Goal:
Evaluate the complete model.

Generate:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC Curve (if applicable)

Grad-CAM samples

Classification Report

Export all figures.