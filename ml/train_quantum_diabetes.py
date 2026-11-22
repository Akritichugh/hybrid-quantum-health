
import os
import json
import copy
import joblib
import numpy as np
import pandas as pd
import pennylane as qml

from pennylane import numpy as pnp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score
)


# ============================================================
# 1. Paths and configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

N_QUBITS = 4
N_LAYERS = 3
HIDDEN_SIZE = 8

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01

PATIENCE = 5
RANDOM_STATE = 42


# ============================================================
# 2. Load and split dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Outcome"]).values.astype(float)
y = df["Outcome"].values.astype(int)

# Keep the test set separate until final evaluation
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# Create validation data from the training portion
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("Dataset split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# ============================================================
# 3. Classical preprocessing
# ============================================================

# Fit preprocessing only on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=N_QUBITS)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Map features to quantum rotation angles
X_train_angles = np.tanh(X_train_pca) * np.pi
X_val_angles = np.tanh(X_val_pca) * np.pi
X_test_angles = np.tanh(X_test_pca) * np.pi


# ============================================================
# 4. Quantum circuit
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(
    dev,
    interface="autograd",
    diff_method="backprop"
)
def quantum_circuit(inputs, quantum_weights):

    # Encode classical features
    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    # Trainable quantum layers
    qml.BasicEntanglerLayers(
        quantum_weights,
        wires=range(N_QUBITS)
    )

    # Measure all qubits
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    ]


# ============================================================
# 5. Hybrid quantum-classical prediction
# ============================================================

def predict_probability(
    quantum_weights,
    head_weights1,
    head_bias1,
    head_weights2,
    head_bias2,
    inputs
):
    # Quantum feature extraction
    measurements = quantum_circuit(
        inputs,
        quantum_weights
    )

    quantum_features = pnp.stack(measurements)

    # Classical nonlinear hidden layer
    hidden = pnp.tanh(
        pnp.dot(quantum_features, head_weights1)
        + head_bias1
    )

    # Classical output layer
    logit = (
        pnp.dot(hidden, head_weights2)
        + head_bias2
    )

    # Keep sigmoid numerically stable
    logit = pnp.clip(logit, -30.0, 30.0)

    probability = 1.0 / (1.0 + pnp.exp(-logit))

    return pnp.clip(
        probability,
        1e-7,
        1.0 - 1e-7
    )


# ============================================================
# 6. Class-weighted binary cross-entropy
# ============================================================

positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

positive_weight = (
    negative_count / max(positive_count, 1)
)

print("\nClass balance:")
print("Positive training samples:", positive_count)
print("Negative training samples:", negative_count)
print("Positive class weight:", positive_weight)


def batch_loss(
    quantum_weights,
    head_weights1,
    head_bias1,
    head_weights2,
    head_bias2,
    batch_X,
    batch_y
):
    losses = []

    for x, label in zip(batch_X, batch_y):

        probability = predict_probability(
            quantum_weights,
            head_weights1,
            head_bias1,
            head_weights2,
            head_bias2,
            x
        )

        # Binary cross-entropy
        loss = -(
            label * pnp.log(probability)
            + (1 - label) * pnp.log(1 - probability)
        )

        # Class weighting
        sample_weight = (
            positive_weight if label == 1 else 1.0
        )

        losses.append(loss * sample_weight)

    return pnp.mean(pnp.stack(losses))


# ============================================================
# 7. Initialize trainable parameters
# ============================================================

np.random.seed(RANDOM_STATE)

quantum_weights = pnp.array(
    np.random.uniform(
        -0.1,
        0.1,
        size=(N_LAYERS, N_QUBITS)
    ),
    requires_grad=True
)

# Classical hidden layer:
# 4 quantum measurements -> 8 hidden neurons
head_weights1 = pnp.array(
    np.random.uniform(
        -0.1,
        0.1,
        size=(N_QUBITS, HIDDEN_SIZE)
    ),
    requires_grad=True
)

head_bias1 = pnp.array(
    np.zeros(HIDDEN_SIZE),
    requires_grad=True
)

# Classical output layer:
# 8 hidden neurons -> 1 probability
head_weights2 = pnp.array(
    np.random.uniform(
        -0.1,
        0.1,
        size=HIDDEN_SIZE
    ),
    requires_grad=True
)

head_bias2 = pnp.array(
    0.0,
    requires_grad=True
)

optimizer = qml.AdamOptimizer(
    stepsize=LEARNING_RATE
)


# ============================================================
# 8. Validation helper
# ============================================================

def get_probabilities(
    quantum_weights,
    head_weights1,
    head_bias1,
    head_weights2,
    head_bias2,
    features
):
    probabilities = []

    for x in features:
        probability = predict_probability(
            quantum_weights,
            head_weights1,
            head_bias1,
            head_weights2,
            head_bias2,
            pnp.array(x)
        )

        probabilities.append(float(probability))

    return np.array(probabilities)


def find_best_threshold(y_true, probabilities):

    best_threshold = 0.5
    best_f1 = -1.0

    # Tune threshold using validation data only
    thresholds = np.arange(0.20, 0.801, 0.01)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# 9. Training with validation and early stopping
# ============================================================

print("\nStarting hybrid quantum-classical training...")
print("Using the full training split.")
print("Early stopping patience:", PATIENCE)

rng = np.random.default_rng(RANDOM_STATE)

best_val_f1 = -1.0
best_threshold = 0.5
best_epoch = 0
patience_counter = 0

best_parameters = None

for epoch in range(EPOCHS):

    # Shuffle all training samples
    indices = rng.permutation(len(X_train_angles))

    epoch_losses = []

    for start in range(0, len(indices), BATCH_SIZE):

        batch_indices = indices[
            start:start + BATCH_SIZE
        ]

        batch_X = pnp.array(
            X_train_angles[batch_indices]
        )

        batch_y = pnp.array(
            y_train[batch_indices]
        )

        # Update all quantum and classical parameters
        (
            quantum_weights,
            head_weights1,
            head_bias1,
            head_weights2,
            head_bias2
        ) = optimizer.step(
            lambda qw, hw1, hb1, hw2, hb2: batch_loss(
                qw,
                hw1,
                hb1,
                hw2,
                hb2,
                batch_X,
                batch_y
            ),
            quantum_weights,
            head_weights1,
            head_bias1,
            head_weights2,
            head_bias2
        )

        current_loss = batch_loss(
            quantum_weights,
            head_weights1,
            head_bias1,
            head_weights2,
            head_bias2,
            batch_X,
            batch_y
        )

        epoch_losses.append(float(current_loss))

    # Evaluate on validation set
    val_probabilities = get_probabilities(
        quantum_weights,
        head_weights1,
        head_bias1,
        head_weights2,
        head_bias2,
        X_val_angles
    )

    val_threshold, val_f1 = find_best_threshold(
        y_val,
        val_probabilities
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {np.mean(epoch_losses):.4f} "
        f"- Validation F1: {val_f1:.4f} "
        f"- Threshold: {val_threshold:.2f}"
    )

    # Save best validation model
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_threshold = val_threshold
        best_epoch = epoch + 1

        best_parameters = {
            "quantum_weights": np.array(quantum_weights).copy(),
            "head_weights1": np.array(head_weights1).copy(),
            "head_bias1": np.array(head_bias1).copy(),
            "head_weights2": np.array(head_weights2).copy(),
            "head_bias2": float(head_bias2)
        }

        patience_counter = 0

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# 10. Restore best model
# ============================================================

print("\nBest epoch:", best_epoch)
print("Best validation F1:", round(best_val_f1, 4))
print("Selected threshold:", round(best_threshold, 2))

quantum_weights = pnp.array(
    best_parameters["quantum_weights"],
    requires_grad=True
)

head_weights1 = pnp.array(
    best_parameters["head_weights1"],
    requires_grad=True
)

head_bias1 = pnp.array(
    best_parameters["head_bias1"],
    requires_grad=True
)

head_weights2 = pnp.array(
    best_parameters["head_weights2"],
    requires_grad=True
)

head_bias2 = pnp.array(
    best_parameters["head_bias2"],
    requires_grad=True
)


# ============================================================
# 11. Final test evaluation
# ============================================================

# Test set is evaluated only after model selection
y_prob = get_probabilities(
    quantum_weights,
    head_weights1,
    head_bias1,
    head_weights2,
    head_bias2,
    X_test_angles
)

y_pred = (
    y_prob >= best_threshold
).astype(int)

metrics = {
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, y_pred)
    ),
    "precision": float(
        precision_score(
            y_test,
            y_pred,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            y_pred,
            zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    ),
    "roc_auc": float(
        roc_auc_score(y_test, y_prob)
    ),
    "threshold": float(best_threshold),
    "best_epoch": int(best_epoch),
    "validation_f1": float(best_val_f1)
}

print("\nTest prediction distribution:")
print("Predicted negative:", np.sum(y_pred == 0))
print("Predicted positive:", np.sum(y_pred == 1))

print("\nTest probability range:")
print("Minimum:", np.min(y_prob))
print("Maximum:", np.max(y_prob))
print("Mean:", np.mean(y_prob))

print("\nHybrid Quantum Diabetes Test Results:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 12. Save model and preprocessing
# ============================================================

model_data = {
    "quantum_weights": np.array(quantum_weights),
    "head_weights1": np.array(head_weights1),
    "head_bias1": np.array(head_bias1),
    "head_weights2": np.array(head_weights2),
    "head_bias2": float(head_bias2),
    "n_qubits": N_QUBITS,
    "n_layers": N_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    "threshold": best_threshold,
    "simulator": "default.qubit"
}

joblib.dump(
    model_data,
    os.path.join(
        MODEL_DIR,
        "quantum_diabetes_model.joblib"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "quantum_diabetes_scaler.joblib"
    )
)

joblib.dump(
    pca,
    os.path.join(
        MODEL_DIR,
        "quantum_diabetes_pca.joblib"
    )
)

metrics_path = os.path.join(
    MODEL_DIR,
    "quantum_diabetes_metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nQuantum model, preprocessing, and metrics saved.")
print("Model:", os.path.join(MODEL_DIR, "quantum_diabetes_model.joblib"))
print("Metrics:", metrics_path)
