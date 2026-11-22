
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ---------------------------------
# Paths
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pneumonia_model.keras"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "chest_xray"
    / "val"
)


# ---------------------------------
# Load model
# ---------------------------------

print("Loading pneumonia model...")
model = tf.keras.models.load_model(MODEL_PATH)


# ---------------------------------
# Load validation dataset
# ---------------------------------

print("Loading validation images...")

dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=(160, 160),
    batch_size=32,
    color_mode="rgb",
    label_mode="binary",
    shuffle=False,
)

print("Class names:", dataset.class_names)

if dataset.class_names != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(
        f"Unexpected class order: {dataset.class_names}"
    )


# ---------------------------------
# Actual labels and probabilities
# ---------------------------------

y_true = np.concatenate(
    [
        labels.numpy().reshape(-1)
        for _, labels in dataset
    ]
).astype(int)

print("Validation images:", len(y_true))

probabilities = model.predict(
    dataset,
    verbose=1
).reshape(-1)


# ---------------------------------
# Evaluate thresholds
# ---------------------------------

thresholds = [
    0.30,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

print("\n===== THRESHOLD COMPARISON =====")
print(
    "Threshold | Accuracy | Precision | Recall | F1 | FP | FN"
)

for threshold in thresholds:
    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print(
        f"{threshold:9.2f} | "
        f"{accuracy * 100:7.2f}% | "
        f"{precision * 100:8.2f}% | "
        f"{recall * 100:6.2f}% | "
        f"{f1 * 100:5.2f}% | "
        f"{fp:3d} | {fn:3d}"
    )


# ---------------------------------
# ROC-AUC
# ---------------------------------

auc = roc_auc_score(
    y_true,
    probabilities
)

print(f"\nValidation ROC-AUC: {auc * 100:.2f}%")
