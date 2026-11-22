from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# -------------------------
# Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pneumonia_model.keras"
)

TEST_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "chest_xray"
    / "test"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "performance_metrics.json"
)


# -------------------------
# Load model
# -------------------------

print("Loading pneumonia model...")

model = tf.keras.models.load_model(MODEL_PATH)


# -------------------------
# Load test dataset
# -------------------------

print("Loading test images...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(160, 160),
    batch_size=32,
    color_mode="rgb",
    label_mode="binary",
    shuffle=False,
)

print("Class names:", test_dataset.class_names)

# Expected alphabetical order:
# NORMAL = 0
# PNEUMONIA = 1

if test_dataset.class_names != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(
        "Unexpected class order. "
        f"Found: {test_dataset.class_names}"
    )


# -------------------------
# Get actual labels
# -------------------------

y_true = np.concatenate(
    [labels.numpy().reshape(-1) for _, labels in test_dataset]
).astype(int)


# -------------------------
# Get model predictions
# -------------------------

print("Evaluating model...")

# Model already contains Rescaling(1/255),
# so do not normalize images again.

probabilities = model.predict(
    test_dataset,
    verbose=1
).reshape(-1)

# Same threshold as your prediction API
y_pred = (probabilities >= 0.5).astype(int)


# -------------------------
# Calculate metrics
# -------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

try:
    auc = roc_auc_score(y_true, probabilities)
except ValueError:
    auc = None

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# -------------------------
# Display results
# -------------------------

print("\n===== PNEUMONIA MODEL PERFORMANCE =====")

print(f"Test images : {len(y_true)}")
print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Precision   : {precision * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"F1-score    : {f1 * 100:.2f}%")

if auc is not None:
    print(f"ROC-AUC     : {auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    )
)


# -------------------------
# Save metrics
# -------------------------

pneumonia_metrics = {
    "accuracy": round(float(accuracy * 100), 2),
    "precision": round(float(precision * 100), 2),
    "recall": round(float(recall * 100), 2),
    "f1_score": round(float(f1 * 100), 2),
    "roc_auc": (
        round(float(auc * 100), 2)
        if auc is not None
        else None
    ),
    "test_samples": int(len(y_true)),
    "confusion_matrix": cm.tolist(),
    "class_names": ["NORMAL", "PNEUMONIA"],
}


# Preserve existing diabetes and heart metrics
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as file:
        all_metrics = json.load(file)
else:
    all_metrics = {}

all_metrics["pneumonia"] = pneumonia_metrics

with open(METRICS_PATH, "w") as file:
    json.dump(
        all_metrics,
        file,
        indent=4
    )

print(f"\nMetrics saved to: {METRICS_PATH}")