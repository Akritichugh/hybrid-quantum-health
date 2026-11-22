
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


# =========================================
# PATHS
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "chest_xray"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pneumonia_model_v2.keras"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "performance_metrics.json"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
THRESHOLD = 0.50


# =========================================
# CHECK FILES
# =========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test folder not found: {TEST_DIR}"
    )


# =========================================
# LOAD MODEL
# =========================================

print("Loading pneumonia V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =========================================
# LOAD UNTOUCHED TEST DATA
# =========================================

print("Loading test images...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=False,
)

class_names = test_dataset.class_names

print("Class names:", class_names)

if class_names != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(
        f"Unexpected class order: {class_names}"
    )


# =========================================
# GET TRUE LABELS
# =========================================

true_labels = np.concatenate(
    [
        labels.numpy().ravel()
        for _, labels in test_dataset
    ]
).astype(int)


# =========================================
# GET MODEL PROBABILITIES
# =========================================

print("Predicting test images...")

probabilities = model.predict(
    test_dataset,
    verbose=1,
).ravel()

predicted_labels = (
    probabilities >= THRESHOLD
).astype(int)


# =========================================
# CALCULATE METRICS
# =========================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

roc_auc = roc_auc_score(
    true_labels,
    probabilities,
)

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1],
)


# =========================================
# PRINT RESULTS
# =========================================

print("\n====================================")
print("PNEUMONIA V2 - TEST PERFORMANCE")
print("====================================")

print("Test images :", len(true_labels))
print("Threshold   :", THRESHOLD)
print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Precision   : {precision * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"F1-score    : {f1 * 100:.2f}%")
print(f"ROC-AUC     : {roc_auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        zero_division=0,
    )
)


# =========================================
# SAVE METRICS
# Keep existing diabetes and heart metrics
# =========================================

if METRICS_PATH.exists():
    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        all_metrics = json.load(file)
else:
    all_metrics = {}


all_metrics["Pneumonia"] = {
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "confusion_matrix": matrix.tolist(),
    "threshold": THRESHOLD,
    "test_images": int(len(true_labels)),
    "model": "pneumonia_model_v2.keras",
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        all_metrics,
        file,
        indent=4,
    )


print("\nMetrics saved to:")
print(METRICS_PATH)
