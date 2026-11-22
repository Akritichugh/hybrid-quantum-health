
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "ml" / "datasets"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

results = {}


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(
            float(accuracy_score(y_test, predictions)), 4
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_score": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    results[model_name] = metrics

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1-score:  {metrics['f1_score'] * 100:.2f}%")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


# =========================================
# DIABETES MODEL
# =========================================

diabetes_df = pd.read_csv(
    DATASET_DIR / "diabetes.csv"
)

diabetes_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

X_diabetes = diabetes_df[diabetes_features]
y_diabetes = diabetes_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X_diabetes,
    y_diabetes,
    test_size=0.2,
    random_state=42,
    stratify=y_diabetes,
)

diabetes_model = joblib.load(
    MODEL_DIR / "diabetes_model.pkl"
)

evaluate_model(
    diabetes_model,
    X_test,
    y_test,
    "Diabetes",
)


# =========================================
# HEART DISEASE MODEL
# =========================================

heart_df = pd.read_csv(
    DATASET_DIR / "heart_disease.csv"
)

heart_features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

X_heart = heart_df[heart_features]
y_heart = heart_df["num"].apply(
    lambda value: 1 if value > 0 else 0
)

X_train, X_test, y_train, y_test = train_test_split(
    X_heart,
    y_heart,
    test_size=0.2,
    random_state=42,
    stratify=y_heart,
)

heart_model = joblib.load(
    MODEL_DIR / "heart_disease_model.joblib"
)

evaluate_model(
    heart_model,
    X_test,
    y_test,
    "Heart Disease",
)


# =========================================
# SAVE METRICS
# =========================================

output_path = MODEL_DIR / "performance_metrics.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nMetrics saved to:")
print(output_path)
